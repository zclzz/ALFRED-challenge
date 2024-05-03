#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#Train BERT on ALFRED data

import random
import time
import torch
from torch import nn
import os
import wandb

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-lr','--learning_rate', type=float, help="learning rate")
parser.add_argument('-model_type','--model_type', type=str, default='bert-base', help="one of roberta-large, roberta-base, bert-base, bert-large, deberta-base, flan-t5-base, flan-t5-large")
parser.add_argument('-task_desc_only','--task_desc_only', action='store_true', help="one of roberta-large, roberta-base, bert-base, bert-large, deberta-base, flan-t5-base, flan-t5-large")
parser.add_argument('--no_appended', action='store_true')


args = parser.parse_args()


##########
#### 1. Data loading and processing
#########

#Set device
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

#Set wandb to track training progress
    
# wandb.init(entity='chaozhoulim', name=f"train_{args.model_type}_lr_{args.learning_rate}", 
#       # Track hyperparameters and run metadata
#       config={
#       "learning_rate": args.learning_rate,
#       "architecture": args.model_type,
#       "epochs": 50,
#       }, project=f"train_{args.model_type}_lr_{args.learning_rate}")

import pickle
directory = 'data/alfred_data/'
# All sets as a dictionary data type with keys ['x', 'y', 's', 'mrecep_targets', 'object_targets', 'parent_targets', 'toggle_targets', 'x_low']
train_set = pickle.load(open('data/alfred_data/train_text_with_ppdl_low_appended.p', 'rb'))
val_set_seen = pickle.load(open('data/alfred_data/val_seen_text_with_ppdl_low_appended.p', 'rb'))
val_set_unseen = pickle.load(open('data/alfred_data/val_unseen_text_with_ppdl_low_appended.p', 'rb'))

#Now process like huggingface
if args.no_appended: # When the --no_appended flag is set, we use the data without the appended ppdl, i.e. the data with only the task description
    # x: data with only task description
    # x_low: data with task description + ppdl
    x_train = train_set['x']; y_train = train_set['y']
else:
    x_train = train_set['x_low']; y_train = train_set['y']
labels = torch.tensor(y_train).to(device)

if args.model_type in ['bert-base', 'bert-large'] : 
    from transformers import BertTokenizer as Tokenizer
    tok_type = args.model_type + '-uncased'
    from transformers import BertForSequenceClassification as BertModel
elif args.model_type in ['roberta-base', 'roberta-large']:
    from transformers import RobertaTokenizer as Tokenizer
    tok_type = args.model_type
    from transformers import RobertaForSequenceClassification as BertModel
elif args.model_type in ['deberta-base']:
    from transformers import AutoTokenizer as DebertaTokenizer
    tok_type = args.model_type
    from transformers import DebertaForSequenceClassification as DebertaModel
elif args.model_type in ['flan-t5-base', 'flan-t5-large']:
    from transformers import T5Tokenizer
    tok_type = args.model_type
    from transformers import T5ForConditionalGeneration as T5Model

if args.model_type in ['bert-base', 'bert-large', 'roberta-base', 'roberta-large']:
    tokenizer = Tokenizer.from_pretrained(tok_type)
    model = BertModel.from_pretrained(tok_type, num_labels=7).to(device)

elif args.model_type in ['deberta-base']: # mismatch of tensors between bert and deberta 
    tokenizer = DebertaTokenizer.from_pretrained('microsoft/deberta-base')
    model = DebertaModel.from_pretrained('microsoft/deberta-base').to(device)

elif args.model_type in ['flan-t5-base', 'flan-t5-large']:
    tokenizer = T5Tokenizer.from_pretrained(f"google/{args.model_type}")
    model = T5Model.from_pretrained(f"google/{args.model_type}").to(device)

encoding = tokenizer(x_train, return_tensors='pt', padding=True, truncation=True)

input_ids = encoding['input_ids'].to(device) #has shape [21025, 37]
attention_mask = encoding['attention_mask'].to(device) #also has shape [21025, 37]


#Get input_ids, attention_mask for val_seen, val_unseen
x_val_seen = val_set_seen['x_low']; y_val_seen = val_set_seen['y']
if args.no_appended:
    x_val_seen = val_set_seen['x']
#Just sample a few from x_val_seen, y_val_seen
random.seed(0)
sampled_val_seen = random.sample(range(len(x_val_seen)), 100)
x_val_seen = [x_val_seen[i] for i in sampled_val_seen]; y_val_seen = [y_val_seen[i] for i in sampled_val_seen]
labels_val_seen = torch.tensor(y_val_seen).to(device)
encoding_v_s = tokenizer(x_val_seen, return_tensors='pt', padding=True, truncation=True)
input_ids_val_seen = encoding_v_s['input_ids'].to(device)
attention_mask_val_seen = encoding_v_s['attention_mask'].to(device) 

x_val_unseen = val_set_unseen['x_low']; y_val_unseen = val_set_unseen['y']
if args.no_appended:
    x_val_unseen = val_set_unseen['x']
random.seed(1)
sampled_val_unseen = random.sample(range(len(x_val_unseen)), 100)
x_val_unseen = [x_val_unseen[i] for i in sampled_val_unseen]; y_val_unseen = [y_val_unseen[i] for i in sampled_val_unseen]
labels_val_unseen = torch.tensor(y_val_unseen).to(device)
encoding_v_us = tokenizer(x_val_unseen, return_tensors='pt', padding=True, truncation=True)
input_ids_val_unseen = encoding_v_us['input_ids'].to(device)
attention_mask_val_unseen = encoding_v_us['attention_mask'].to(device)



##########
#### 2. Do training
#########

model.train()
from transformers import AdamW
optimizer = AdamW(model.parameters(), lr=args.learning_rate)
N = 16 #batch size
if args.no_appended:
    N=64
save_folder_name = args.model_type + '_lr_' + str(args.learning_rate) +'/'
if args.no_appended:
    super_folder = 'saved_models_noappended/'
else:
    super_folder = 'saved_models/'
if not os.path.exists(super_folder + save_folder_name):
    os.makedirs(super_folder + save_folder_name)


def accuracy(y_pred, y_batch):
    #y_pred has shape [batch, no_classes]
    maxed = torch.max(y_pred, 1)
    y_hat = maxed.indices
    num_accurate = torch.sum((y_hat == y_batch).float())
    train_accuracy = num_accurate/ y_hat.shape[0]
    return train_accuracy.item()

def accurate_total(y_pred, y_batch):
    #y_pred has shape [batch, no_classes]
    maxed = torch.max(y_pred, 1)
    y_hat = maxed.indices
    num_accurate = torch.sum((y_hat == y_batch).float())
    return num_accurate
start_train_time = time.time()
for t in range(50):
    model.train()
    avg_training_loss = 0.0
    training_acc = 0.0
    for b in range(int(input_ids.shape[0]/N)):
        input_ids_batch = input_ids[N*b:N*(b+1)].to(device) #si
        labels_batch = labels[N*b:N*(b+1)].to(device) #selects a subset of gt labels from the training set
        attention_mask_batch = attention_mask[N*b:N*(b+1)].to(device)

        print('labels_batch:', labels_batch)
        print('labels_batch shape:', labels_batch.shape)
        print('labels_batch (-1):', labels_batch.view(-1).shape)
        print('labels_batch (1, -1) shape:', labels_batch.view(1, -1).shape)
        print('input_ids_batch:', input_ids_batch)
        print('input_ids_batch shape:', input_ids_batch.shape)
        print('attention_mask:', attention_mask)
        print('attention_mask shape:', attention_mask.shape)
        print('attention_mask_batch shape:', attention_mask_batch.shape)
        optimizer.zero_grad()
        #forward pass
        if args.model_type in ['bert-base', 'bert-large', 'roberta-base', 'roberta-large']:
            outputs = model(input_ids_batch, attention_mask=attention_mask_batch, labels=labels_batch.view(1,-1)) # reshapes labels_batch to be [1, batch_size], resulting in 2D tensor
        elif args.model_type in ['deberta-base', 'deberta-large']:
            outputs = model(input_ids_batch, attention_mask=attention_mask_batch, labels=labels_batch.view(-1)) # reshapes labels_batch to be [batch_size], resulting in 1D tensor
        elif args.model_type in ['flan-t5-base', 'flan-t5-large']:
            outputs = model(input_ids_batch, attention_mask=attention_mask_batch, labels=labels_batch.view(-1))
        # outputs = model(input_ids_batch, attention_mask=attention_mask_batch, labels=labels_batch.view(-1)) #for bert
        # outputs = model(input_ids_batch, attention_mask=attention_mask_batch, labels=labels_batch.view(1,-1)) #for deberta
        # outputs = model(input_ids_batch, attention_mask=attention_mask_batch, labels=input_ids_batch) #for flan-t5

        loss = outputs.loss
        num_acc = accurate_total(y_pred=outputs.logits, y_batch=labels_batch.view(-1))
        loss.backward()
        optimizer.step()
        #
        avg_training_loss += loss
        training_acc += num_acc 
    avg_training_loss *= 1/int(input_ids.shape[0]/N)
    #Print & Evaluate
    if t % 1 ==0:
        with torch.no_grad():
            model.eval()
            print("loss at step ", t, " : ", loss.item())
            print("training accuracy: ", training_acc/input_ids.shape[0])
            #evaluate
            outputs_val_seen = model(input_ids_val_seen, attention_mask=attention_mask_val_seen)
            val_seen_acc = accuracy(y_pred=outputs_val_seen.logits, y_batch=labels_val_seen.view(-1))
            print("validation accuracy (seen): ", accuracy(y_pred=outputs_val_seen.logits, y_batch=labels_val_seen.view(-1)))
            del outputs_val_seen
            outputs_val_unseen = model(input_ids_val_unseen, attention_mask=attention_mask_val_unseen)
            val_unseen_acc = accuracy(y_pred=outputs_val_unseen.logits, y_batch=labels_val_unseen.view(-1))
            print("validation accuracy (unseen): ", accuracy(y_pred=outputs_val_unseen.logits, y_batch=labels_val_unseen.view(-1)))
            wandb.log({'training_loss': loss.item(), 'train_acc': training_acc/input_ids.shape[0], 'val_seen_acc': val_seen_acc, 'val_unseen_acc': val_unseen_acc})

            model_name = 'epoch_' + str(t) + '.pt'
            torch.save(model.state_dict(), super_folder + save_folder_name + model_name)
            print("model saved")
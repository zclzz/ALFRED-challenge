#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random
import time
import torch
from torch import nn
import os
import glob
from collections import OrderedDict
import argparse
import wandb
parser = argparse.ArgumentParser()
parser.add_argument('-lr','--learning_rate', type=float, default=1e-5, help="learning rate")
parser.add_argument('-s','--seed', type=int, default=0, help="seed")
parser.add_argument('-d','--decay', type=float, default=0.5, help="template") 
parser.add_argument('-dt','--decay_term', type=int, default=5, help="template")
parser.add_argument('-v','--verbose', type=int, default=0, help="print training output")
parser.add_argument('-load','--load', type=str, default='', help="one of roberta-large, roberta-base, bert-base, bert-large")
parser.add_argument('-no_divided_label','--no_divided_label', action='store_true')
parser.add_argument('--no_appended', action='store_true') # if flag is present, return true
# parser.add_argument('-es', '--early_stopping', type=int, default=5, help="early stopping patience")


args = parser.parse_args()

#Set device
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# wandb.init(entity='chaozhoulim', name=f"train_t5-base_CondGen_Model_all_parameters", 
#       # Track hyperparameters and run metadata
#       config={
#       "learning_rate": args.learning_rate,
#       "architecture": 't5-base',
#       "epochs": 50,
#       }, project='train_' + 't5-base_CondGen_Model' + 'all_parameters')

import pickle
import numpy as np

template_by_label = pickle.load(open('data/alfred_data/alfred_dicts/template_by_label.p', 'rb'))
train_set = pickle.load(open('data/alfred_data/train_text_with_ppdl_low_appended.p', 'rb'))
val_set_seen = pickle.load(open('data/alfred_data/val_seen_text_with_ppdl_low_appended.p', 'rb'))
val_set_unseen = pickle.load(open('data/alfred_data/val_unseen_text_with_ppdl_low_appended.p', 'rb'))

obj2idx = pickle.load(open('data/alfred_data/alfred_dicts/obj2idx.p', 'rb'))
idx2obj = pickle.load(open('data/alfred_data/alfred_dicts/idx2obj.p', 'rb'))
idx2recep = pickle.load(open('data/alfred_data/alfred_dicts/idx2recep.p', 'rb'))
recep2idx = pickle.load(open('data/alfred_data/alfred_dicts/recep2idx.p', 'rb'))
toggle2idx = pickle.load(open('data/alfred_data/alfred_dicts/toggle2idx.p', 'rb'))

train_prefix_high = train_set['x_prefix']
train_prefix_low = train_set['x_low_prefix']
label_parameters = train_set['parameters']

if args.no_appended:
    train_data = train_prefix_high
else:
    train_data = train_prefix_low

# print(train_data)

from transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained('t5-base')
model = T5ForConditionalGeneration.from_pretrained('t5-base').to(device)

####################################################
## 1. Prepare Data
####################################################

input_ids = tokenizer(train_data, return_tensors='pt', padding='longest', truncation=True).input_ids.to(device) # tokenized input
labels = tokenizer(label_parameters, return_tensors='pt', padding='longest', truncation=True).input_ids.to(device) # tokenized labels
labels[labels == tokenizer.pad_token_id] = -100

print("input_ids shape", input_ids.shape)
print("labels shape", labels.shape)

# do the same for validation seen set

val_seen_prefix_high = val_set_seen['x_prefix']
val_seen_prefix_low = val_set_seen['x_low_prefix']
val_seen_label_parameters = val_set_seen['parameters']

if args.no_appended:
    val_seen_data = val_seen_prefix_high
else:
    val_seen_data = val_seen_prefix_low

from torch.utils.data import DataLoader, TensorDataset

input_ids_val_seen = tokenizer(val_seen_data, return_tensors='pt', padding='longest', truncation=True).input_ids.to(device) # tokenized val seen input
labels_val_seen = tokenizer(val_seen_label_parameters, return_tensors='pt', padding='longest', truncation=True).input_ids.to(device) # tokenized labels
labels_val_seen[labels_val_seen == tokenizer.pad_token_id] = -100

val_seen_dataset = TensorDataset(input_ids_val_seen, labels_val_seen)
val_seen_loader = DataLoader(val_seen_dataset, batch_size=100)  # Batch size set to 100 for validation

####################################################
## 2. Do training
####################################################

print('Model initialized successfully')
model.train()
from transformers import AdamW
learning_rate = args.learning_rate
optimizer = AdamW(model.parameters(), lr=learning_rate)

N = 32 # batch size

def accuracy(outputs_batch, labels_batch):
    y_hat = outputs_batch.argmax(dim=-1)
    batch_acc = (y_hat == labels_batch).float().mean()
    return batch_acc # returns the accuracy of the batch

if args.no_appended:
    super_folder = 'saved_models_noappended/'
else:
    super_folder = 'saved_models/'

save_folder_name = 'all_parameters/t5-base_CondGenModel_lr_' + str(args.learning_rate) + 'seed_' + str(args.seed) + 'decay_' + str(args.decay) +'/'
if not os.path.exists(super_folder+'argument_models/' + save_folder_name):
    os.makedirs(super_folder+'argument_models/' + save_folder_name)
    
accuracy_dictionary = {'training_loss': [], 'training':[], 'val_seen_loss': [],'val_seen_acc':[]}

# best_val_acc = 0.0
# epochs_no_improve = 0
# early_stop = args.early_stopping

for t in range(50):
    model.train()
    if t>0 and (t+1)%args.decay_term ==0:
        learning_rate *= args.decay
        optimizer = AdamW(model.parameters(), lr=learning_rate)
    avg_training_loss = 0.0
    avg_training_acc = 0.0
    for b in range(int(input_ids.shape[0]/N)):
        print('input_ids shape: ', input_ids.shape)
        input_ids_batch = input_ids[N*b:N*(b+1)].to(device) # has shape [64, 475]
        print('input_ids_batch shape: ', input_ids_batch.shape)
        labels_batch = labels[N*b:N*(b+1)].to(device)
        optimizer.zero_grad() # clear gradients
        print('labels_batch shape: ', labels_batch.shape)
        print('labels_batch: ', labels_batch)

        #forward pass
        outputs = model(input_ids_batch, labels=labels_batch)
        accuracy_batch = accuracy(outputs.logits, labels_batch)
        avg_training_acc += accuracy_batch.item()

        loss = outputs.loss
        avg_training_loss += loss.item()
        #if t ==0:
        #    print("loss at step ", t, " : ", loss.item())
        loss.backward()
        optimizer.step()

    avg_training_loss *= 1/int(input_ids.shape[0]/N)
    avg_training_acc *= 1/int(input_ids.shape[0]/N)
    accuracy_dictionary['training_loss'].append(avg_training_loss)
    accuracy_dictionary['training'].append(avg_training_acc)

    #Print & Evaluate
    if args.verbose:
        print("loss at step ", t, " : ", loss.item())
        print("training accuracy: ", avg_training_acc)
    #evaluate
    model.eval()
    avg_val_seen_loss = 0.0
    avg_val_seen_acc = 0.0

    with torch.no_grad():
        for batch in val_seen_loader:
            val_seen_input_ids_batch, val_seen_labels_batch = batch

            outputs_val_seen = model(val_seen_input_ids_batch, labels=val_seen_labels_batch)
            val_accuracy_batch = accuracy(outputs_val_seen.logits, val_seen_labels_batch)
            avg_val_seen_acc += val_accuracy_batch.item()

            val_seen_loss = outputs_val_seen.loss
            avg_val_seen_loss += val_seen_loss.item()

    
        model_name = 'epoch_' + str(t) + '.pt'
        torch.save(model.state_dict(), super_folder+'argument_models/' + save_folder_name + model_name)

        avg_val_seen_acc *= 1/len(val_seen_loader)
        avg_val_seen_loss *= 1/len(val_seen_loader)
        accuracy_dictionary['val_seen_acc'].append(avg_val_seen_acc)
        accuracy_dictionary['val_seen_loss'].append(avg_val_seen_loss)
        if args.verbose:
            print("validation (seen) accuracy: ", avg_val_seen_acc)
        del outputs_val_seen
        
        wandb.log({'training_loss': loss.item(), 'train_acc': avg_training_acc, 'val_seen_loss': avg_val_seen_loss, 'val_seen_acc': avg_val_seen_acc})
        # Check if validation accuracy improved

        # early stopping
        # if accuracy_dictionary['val_seen_acc'][-1] > best_val_acc:
        #     best_val_acc = accuracy_dictionary['val_seen_acc'][-1]
        #     epochs_no_improve = 0
        #     # Save the best model
        #     best_model_path = super_folder+'argument_models/' + save_folder_name + 'best_model.pt'
        #     torch.save(model.state_dict(), best_model_path)
        # else:
        #     epochs_no_improve += 1 # keep track of epochs with no improvement

        # # Early stopping
        # if epochs_no_improve == early_stop:
        #     print(f"Early stopping triggered. No improvement in validation accuracy for {early_stop} consecutive epochs.")
        #     break

    
#Get the highest accuracy and delete the rest 
highest_test = np.argwhere(accuracy_dictionary['val_seen_acc'] == np.amax(accuracy_dictionary['val_seen_acc']))
highest_test = highest_test.flatten().tolist()

training_acc_highest_h = np.argmax([accuracy_dictionary['training'][h] for h in highest_test])
best_t = highest_test[training_acc_highest_h]
print("The best model is 'epoch_", str(best_t)+ '.pt')

#Delete every model except for best_t 
file_path = super_folder+'argument_models/' + save_folder_name + "epoch_" + str(best_t) + ".pt"
if os.path.isfile(file_path):
    for CleanUp in glob.glob(super_folder+'argument_models/' + save_folder_name + '*.pt'):
        if not CleanUp.endswith(file_path):    
            os.remove(CleanUp)

#Save training/ test accuracy dictionary in a txt file
f = open(super_folder+ 'argument_models/' + save_folder_name +  "training_log.txt", "w")
for t in range(50):
    if t == best_t:
        f.write("===========================================\n")
    f.write("Epoch " + str(t) + "\n")
    f.write("training loss: " + str(accuracy_dictionary['training_loss'][t]) + "\n")
    f.write("training accuracy: " + str(accuracy_dictionary['training'][t]) + "\n")
    f.write("validation (seen) accuracy: " + str(accuracy_dictionary['val_seen_acc'][t]) + "\n")
    if t == best_t:
        f.write("===========================================")
f.close()

#Print training/ test accuracy for t
print("Saved and finished")


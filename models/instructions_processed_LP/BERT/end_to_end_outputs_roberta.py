'''
In summary, this script is part of a system that takes natural language instructions 
and predicts various parameters and labels related to robot control tasks based on 
a pre-trained Roberta model. It uses the templates obtained from the base model to guide
the extraction of specific arguments and labels for these tasks.
'''
#Run base model (into templates) and then extract arguments

import random
import time
import torch
from torch import nn
import pickle
import glob
import argparse
import os

os.environ["CURL_CA_BUNDLE"] = ""

parser = argparse.ArgumentParser()
parser.add_argument('-sp','--split', type=str, choices=['val_unseen', 'val_seen', 'tests_seen', 'tests_unseen', 'others'], required=True)
parser.add_argument('-m','--model_saved_folder_name', type=str, required=True) # folder where the model is saved
parser.add_argument('-o','--output_name', type=str, required=True) # save the output as a pickle file to this name
parser.add_argument('--no_appended', action='store_true')

args = parser.parse_args()


def accuracy(y_pred, y_batch):
    #y_pred has shape [batch, no_classes]
    maxed = torch.max(y_pred, 1)
    y_hat = maxed.indices
    num_accurate = torch.sum((y_hat == y_batch).long())
    train_accuracy = num_accurate/ y_hat.shape[0]
    return train_accuracy.item()

def accurate_both(y_pred1, y_batch1, y_pred2, y_batch2):
    #
    maxed1 = torch.max(y_pred1, 1)
    y_hat1 = maxed1.indices
    #
    maxed2 = torch.max(y_pred2, 1)
    y_hat2 = maxed2.indices
    #
    num_both_accurate = torch.sum((y_hat1 == y_batch1).long() * (y_hat2 == y_batch2).long())
    train_accuracy = num_both_accurate/ y_hat1.shape[0]
    return train_accuracy.item()


#Load data
import pickle
split = args.split
# template_by_label is a dictionary that maps a label to a template
template_by_label = pickle.load(open('data/alfred_data/alfred_dicts/correct_template_by_label_ppdl.p', 'rb'))
# new_labels_dict is a dictionary that maps a label to a new label
new_labels_dict = pickle.load(open('data/alfred_data/alfred_dicts/correct_labels_dict_ppdl.p', 'rb'))
if split == 'val_unseen':
    val_set_unseen_or_seen = pickle.load(open('data/alfred_data/'+ 'val_unseen_text_with_ppdl_low_appended.p', 'rb'))
elif split == 'val_seen':
    val_set_unseen_or_seen = pickle.load(open('data/alfred_data/'+ 'val_seen_text_with_ppdl_low_appended.p', 'rb'))
else:
    print('Please use one of the valid splits: val_unseen, val_seen only')
    exit()
    # val_set_unseen = pickle.load(open('data/alfred_data/'+ 'pan_example.p', 'rb'))

# val_set_unseen = pickle.load(open('data/input_lang.p', 'rb'))[args.split]

obj2idx = pickle.load(open('data/alfred_data/alfred_dicts/obj2idx.p', 'rb')) # gets the index of each object
recep2idx = pickle.load(open('data/alfred_data/alfred_dicts/recep2idx.p', 'rb')) # gets the index of each receptacle
toggle2idx = pickle.load(open('data/alfred_data/alfred_dicts/toggle2idx.p', 'rb')) # gets the index of each toggle

idx2obj = {v:k for k, v in obj2idx.items()} #only glassbottle should remain
idx2recep = {v:k for k, v in recep2idx.items()} # re-format the dict and sorts the target according to the index
idx2toggle = {v:k for k, v in toggle2idx.items()} # eg. {0: 'Candle', 1: 'CellPhone', ...}

#These are based on new labels
task_to_label_mapping = {'mrecep_target':[1], 'object_target': [0,1,2,3,4,5,6],\
                         'parent_target':[0,1,2,3,4,6], 'toggle_target': [5],
                         'sliced':[0,1,2,3,4,5,6]}

#Set device
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


save_folder_name = args.model_saved_folder_name
#Base model
# The Base model is the model that outputs the template describing the task. For example
# the template for the task "Put the apple on the table" is "Put the [object_target] on the [parent_target]"
from transformers import RobertaForSequenceClassification
base_model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=7).to(device)

base_model_name = 'roberta_base_e49.pt'
base_model.load_state_dict(torch.load(os.path.join(save_folder_name, base_model_name)))
base_model.eval()
# print('base model:', base_model.eval())
#Run data through base model and get label
# label means the template for example "Put the [object_target] on the [parent_target]" 
from transformers import RobertaTokenizer
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

#Put validation data into the base_model
x_val_seen = val_set_unseen_or_seen['x_low'] #; y_val_seen = val_set_seen['y']
if args.no_appended: # if no appended flag is set, we will make predictions from only high-level language
    x_val_seen = val_set_unseen_or_seen['x']

encoding_v_s = tokenizer(x_val_seen, return_tensors='pt', padding=True, truncation=True)
input_ids_val_seen = encoding_v_s['input_ids'].to(device)
attention_mask_val_seen = encoding_v_s['attention_mask'].to(device) 

N = 8
y_hat_list_vs = []
if input_ids_val_seen.shape[0]%N!=0:
    until = int(input_ids_val_seen.shape[0]/N)+1
else:
    until = int(input_ids_val_seen.shape[0]/N)
for b in range(until):
    input_ids_batch = input_ids_val_seen[N*b:N*(b+1)].to(device)
    attention_mask_batch = attention_mask_val_seen[N*b:N*(b+1)].to(device)    
    outputs = base_model(input_ids_batch, attention_mask=attention_mask_batch)
    predicted_templates = torch.max(outputs.logits, 1).indices
    y_hat_list_vs += predicted_templates.cpu().numpy().tolist()

del outputs
del base_model
vs_idx2predicted_label = {i:y for i, y in enumerate(y_hat_list_vs)}
# print(vs_idx2predicted_label, 'vs_idx2predicted_label')
#Now extract the arguments 
# Extract the arguments for example, for the task "Put the apple on the table", the arguments are
# object_target = apple and parent_target = table
def get_prediction(classifier, N, input_ids, attention_mask):
    y_hat_list = []
    for b in range(int(input_ids.shape[0]/N)+1):
        if b!=int(input_ids.shape[0]/N):
            input_ids_batch = input_ids[N*b:N*(b+1)].to(device)
        else:
            input_ids_batch = input_ids[N*b:].to(device)
        attention_mask_batch = attention_mask[N*b:N*(b+1)].to(device)
        
        #outputs = base_model(input_ids_batch, attention_mask=attention_mask_batch, labels=labels_batch.view(1,-1))
        outputs = classifier(input_ids_batch, attention_mask=attention_mask_batch)
        predicted_templates = torch.max(outputs.logits, 1).indices
        del outputs
        y_hat_list += predicted_templates.cpu().numpy().tolist()
    return y_hat_list

x_val_seen_p = [str(int(vs_idx2predicted_label[i])) + ' ' + x for i, x in enumerate(x_val_seen)]
encoding_v_s = tokenizer(x_val_seen_p, return_tensors='pt', padding=True, truncation=True)
input_ids_val_seen = encoding_v_s['input_ids'].to(device)
attention_mask_val_seen = encoding_v_s['attention_mask'].to(device) 

from transformers import BertForSequenceClassification
parent_target_classifier = BertForSequenceClassification.from_pretrained('roberta-base', num_labels=len(recep2idx)).to(device)     
parent_target_classifier.load_state_dict(torch.load(os.path.join(save_folder_name, 'parent_roberta.pt')))    
parent_outputs_hat = get_prediction(parent_target_classifier, 9, input_ids_val_seen, attention_mask_val_seen)
del parent_target_classifier

object_target_classifier = BertForSequenceClassification.from_pretrained('roberta-base', num_labels=len(obj2idx)).to(device)     
object_target_classifier.load_state_dict(torch.load(os.path.join(save_folder_name, 'objects_roberta.pt')))
object_outputs_hat = get_prediction(object_target_classifier, 9, input_ids_val_seen, attention_mask_val_seen)
del object_target_classifier

sliced_target_classifier = BertForSequenceClassification.from_pretrained('roberta-base', num_labels=2).to(device)     
sliced_target_classifier.load_state_dict(torch.load(os.path.join(save_folder_name, 'sliced_roberta.pt')))
sliced_outputs_hat = get_prediction(sliced_target_classifier, 9, input_ids_val_seen, attention_mask_val_seen)
del sliced_target_classifier

mrecep_target_classifier = BertForSequenceClassification.from_pretrained('roberta-base', num_labels=len(obj2idx)).to(device)     
mrecep_target_classifier.load_state_dict(torch.load(os.path.join(save_folder_name, 'mrecep_roberta.pt')))
mrecep_outputs_hat = get_prediction(mrecep_target_classifier, 9, input_ids_val_seen, attention_mask_val_seen)
del mrecep_target_classifier

# print('parent_outputs_hat:', parent_outputs_hat, 'object_outputs_hat:', object_outputs_hat, 'sliced_outputs_hat:', sliced_outputs_hat, 'mrecep_outputs_hat:', mrecep_outputs_hat)
instructions= val_set_unseen_or_seen['x_low']
instruction2_params_test_unseen = {}
for i, instruction in enumerate(instructions):
    # print(vs_idx2predicted_label, 'vs_idx2predicted_label')
    task_type = vs_idx2predicted_label[i]
    
    object_target = idx2obj[object_outputs_hat[i]]
    if parent_outputs_hat == None:
        parent_target = None
    else:
        parent_target = idx2recep[parent_outputs_hat[i]]
    if mrecep_outputs_hat == None:
        # print('mrecep_outputs_hat is None', mrecep_outputs_hat)
        mrecep_target = None
    else:
        mrecep_target = idx2obj[mrecep_outputs_hat[i]]
        # print('mrecep_outputs_hat is not None', mrecep_outputs_hat)
        # print('mrecep_target', mrecep_target)
    sliced_target = sliced_outputs_hat[i]

    if task_type == 5: # if task_type is 'look_at_obj_in_light'
        parent_target = None
    if task_type !=1: # if task_type is not 'pick_and_place_with_moveable_recep'
        mrecep_target = None

    instruction2_params_test_unseen[instruction] = {'task_type': task_type, 'mrecep_target': mrecep_target,\
                                                      'object_target': object_target,\
                                                      'parent_target': parent_target,\
                                                  'sliced': sliced_target}
print(val_set_unseen_or_seen['x'])
print(instruction2_params_test_unseen)
pickle.dump(instruction2_params_test_unseen, open(args.output_name + ".p", "wb"))



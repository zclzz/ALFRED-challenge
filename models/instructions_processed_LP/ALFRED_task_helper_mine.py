#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pickle
import alfred_utils.gen.constants as constants
import string

exclude = set(string.punctuation)
task_type_dict = {2: 'pick_and_place_simple',
 5: 'look_at_obj_in_light',
 1: 'pick_and_place_with_movable_recep',
 3: 'pick_two_obj_and_place',
 6: 'pick_clean_then_place_in_recep',
 4: 'pick_heat_then_place_in_recep',
 0: 'pick_cool_then_place_in_recep'}

def read_test_dict(test, language_granularity, unseen):
    '''
    This function reads the test dictionary for the test set.

    Parameters:
    - test: A flag indicating if the test set is being used.
    - language_granularity: The granularity of the language (gt, high, high_low).
    - unseen: A flag indicating if the test set is unseen.

    Returns:
    - A dictionary containing task-specific information.
    '''
    # list_of_test_dict = []
    # print(test, language_granularity, unseen, "testing")
    split_name = "test" if test else "val"
    split_type = "unseen" if unseen else "seen"

    granularity_map = {"gt": "gt", "high": "noappended", "high_low": "appended"}
    granularity = granularity_map[language_granularity]

    # print(split_name, split_type, granularity, "testing")
    # dict_fname = f"models/instructions_processed_LP/instruction2_params_{split_name}_{split_type}_{granularity}.p"
    # dict_fname_pred1 = f"models/instructions_processed_LP/val_unseen_T5_All_{granularity} (actual).p"
    # dict_fname_pred1 = f"models/instructions_processed_LP/val_unseen_T5_All_{granularity}_pred1.p"
    dict_fname_pred1 = f"models/instructions_processed_LP/{split_name}_{split_type}_T5_All_{granularity}_pred1.p"
    # list_of_test_dict.append(pickle.load(open(dict_fname_pred1, "rb")))
    # dict_fname_pred2 = f"models/instructions_processed_LP/val_unseen_T5_All_{granularity}_pred2.p"
    # list_of_test_dict.append(pickle.load(open(dict_fname_pred2, "rb")))
    # dict_fname_pred3 = f"models/instructions_processed_LP/val_unseen_T5_All_{granularity}_pred3.p"
    # list_of_test_dict.append(pickle.load(open(dict_fname_pred3, "rb")))
    # dict_fname_pred4 = f"models/instructions_processed_LP/val_unseen_T5_All_{granularity}_pred4.p"
    # list_of_test_dict.append(pickle.load(open(dict_fname_pred4, "rb")))
    # dict_fname_pred5 = f"models/instructions_processed_LP/val_unseen_T5_All_{granularity}_pred5.p"
    # list_of_test_dict.append(pickle.load(open(dict_fname_pred5, "rb")))
    # print(pickle.load(open(dict_fname, "rb")), "testing")
    return pickle.load(open(dict_fname_pred1, "rb"))
    # return list_of_test_dict

# def get_other_test_dict(test, language_granularity, unseen):
#     split_name = "test" if test else "val"
#     split_type = "unseen" if unseen else "seen"

#     granularity_map = {"gt": "gt", "high": "noappended", "high_low": "appended"}
#     granularity = granularity_map[language_granularity]

#     next_list_of_test_dict = []
#     dict_fname_pred2 = f"models/instructions_processed_LP/val_unseen_T5_All_{granularity}_pred2.p"
#     next_list_of_test_dict.append(pickle.load(open(dict_fname_pred2, "rb")))
#     dict_fname_pred3 = f"models/instructions_processed_LP/val_unseen_T5_All_{granularity}_pred3.p"
#     next_list_of_test_dict.append(pickle.load(open(dict_fname_pred3, "rb")))
#     dict_fname_pred4 = f"models/instructions_processed_LP/val_unseen_T5_All_{granularity}_pred4.p"
#     next_list_of_test_dict.append(pickle.load(open(dict_fname_pred4, "rb")))
#     dict_fname_pred5 = f"models/instructions_processed_LP/val_unseen_T5_All_{granularity}_pred5.p"
#     next_list_of_test_dict.append(pickle.load(open(dict_fname_pred5, "rb")))
#     return next_list_of_test_dict

def get_other_test_dict(test, language_granularity, unseen):
    objects = {'AlarmClock': 0, 'Apple': 1, 'AppleSliced': 2, 'BaseballBat': 3, 'BasketBall': 4, 'Book': 5, 'Bowl': 6, 'Box': 7, 'Bread': 8, 'BreadSliced': 9, 'ButterKnife': 10, 'CD': 11, 'Candle': 12, 'CellPhone': 13, 'Cloth': 14, 'CreditCard': 15, 'Cup': 16, 'DeskLamp': 17, 'DishSponge': 18, 'Egg': 19, 'Faucet': 20, 'FloorLamp': 21, 'Fork': 22, 'Glassbottle': 23, 'HandTowel': 24, 'HousePlant': 25, 'Kettle': 26, 'KeyChain': 27, 'Knife': 28, 'Ladle': 29, 'Laptop': 30, 'LaundryHamper': 31, 'Lettuce': 32, 'LettuceSliced': 33, 'LightSwitch': 34, 'Mug': 35, 'Newspaper': 36,
            'Pan': 37, 'PaperTowel': 38, 'PaperTowelRoll': 39, 'Pen': 40, 'Pencil': 41, 'PepperShaker': 42, 'Pillow': 43, 'Plate': 44, 'Plunger': 45, 'Pot': 46, 'Potato': 47, 'PotatoSliced': 48, 'RemoteControl': 49, 'SaltShaker': 50, 'ScrubBrush': 51, 'ShowerDoor': 52, 'SoapBar': 53, 'SoapBottle': 54, 'Spatula': 55, 'Spoon': 56, 'SprayBottle': 57, 'Statue': 58, 'StoveKnob': 59, 'TeddyBear': 60, 'Television': 61, 'TennisRacket': 62, 'TissueBox': 63, 'ToiletPaper': 64, 'ToiletPaperHanger':65, 'ToiletPaperRoll': 66, 'Tomato': 67, 'TomatoSliced': 68, 'Towel': 69, 'Vase': 70, 'Watch': 71, 'WateringCan': 72, 'WineBottle': 73, 'None': 74}
    objects = list(objects.keys())
    # objects.append(None)
    receptacles = {'ArmChair': 0, 'BathtubBasin': 1, 'Bed': 2, 'Cabinet': 3, 'Cart': 4, 'CoffeeMachine': 5, 'CoffeeTable': 6, 'CounterTop': 7, 'Desk': 8, 'DiningTable': 9, 'Drawer': 10,
                        'Dresser': 11, 'Fridge': 12, 'GarbageCan': 13, 'Microwave': 14, 'Ottoman': 15, 'Safe': 16, 'Shelf': 17, 'SideTable': 18, 'SinkBasin': 19, 'Sofa': 20, 'StoveBurner': 21, 'TVStand': 22, 'Toilet': 23, 'None': 24}
    receptacles = list(receptacles.keys())
    receptacles.append(None)

    split_name = "test" if test else "val"
    split_type = "unseen" if unseen else "seen"

    granularity_map = {"gt": "gt", "high": "noappended", "high_low": "appended"}
    granularity = granularity_map[language_granularity]

    next_list_of_test_dict = []
    dict_fname_pred1 = f"models/instructions_processed_LP/{split_name}_{split_type}_T5_All_{granularity}_pred1.p"
    pred1 = pickle.load(open(dict_fname_pred1, "rb")) # first prediction here to replace 

    dict_fname_pred2 = f"models/instructions_processed_LP/{split_name}_{split_type}_T5_All_{granularity}_pred2.p"
    pred2 = pickle.load(open(dict_fname_pred2, "rb"))
    dict_fname_pred3 = f"models/instructions_processed_LP/{split_name}_{split_type}_T5_All_{granularity}_pred3.p"
    pred3 = pickle.load(open(dict_fname_pred3, "rb"))
    dict_fname_pred4 = f"models/instructions_processed_LP/{split_name}_{split_type}_T5_All_{granularity}_pred4.p"
    pred4 = pickle.load(open(dict_fname_pred4, "rb"))
    dict_fname_pred5 = f"models/instructions_processed_LP/{split_name}_{split_type}_T5_All_{granularity}_pred5.p"
    pred5 = pickle.load(open(dict_fname_pred5, "rb"))

    '''
    For our subsequent predictions. there are a few things we must consider:
    1. The object_target, mrecep_target, and parent_target must be in the original dictionary, this becomes less likely with the increase in number of beams
    2. It doesnt make sense if the object target is sliced or None, every task has to have a target object which is unsliced
    '''

    # pred2_dict, pred3_dict, pred4_dict, pred5_dict = {}, {}, {}, {}
    for keys in pred2.keys(): # ensures that subsequent dictionaries have the objects that are in the original dictionary
        if (pred2[keys]['object_target'] is None) or not((pred2[keys]['object_target'] in objects) and (pred2[keys]['mrecep_target'] in objects) and (pred2[keys]['parent_target'] in receptacles)) or (pred2[keys]['object_target'] is not None and 'Sliced' in pred2[keys]['object_target']):
            pred2[keys] = pred1[keys] # if the object is not in the original dictionary, then it is replaced with the predictions from the first dictionary
        if (pred3[keys]['object_target'] is None) or not((pred3[keys]['object_target'] in objects) and (pred3[keys]['mrecep_target'] in objects) and (pred3[keys]['parent_target'] in receptacles)) or (pred3[keys]['object_target'] is not None and 'Sliced' in pred3[keys]['object_target']):
            pred3[keys] = pred1[keys]
        if (pred4[keys]['object_target'] is None) or not((pred4[keys]['object_target'] in objects) and (pred4[keys]['mrecep_target'] in objects) and (pred4[keys]['parent_target'] in receptacles)) or (pred4[keys]['object_target'] is not None and 'Sliced' in pred4[keys]['object_target']):
            pred4[keys] = pred1[keys]
        if (pred5[keys]['object_target'] is None) or not((pred5[keys]['object_target'] in objects) and (pred5[keys]['mrecep_target'] in objects) and (pred5[keys]['parent_target'] in receptacles)) or (pred5[keys]['object_target'] is not None and 'Sliced' in pred5[keys]['object_target']):
            pred5[keys] = pred1[keys]
    # next_list_of_test_dict.append(pickle.load(open(dict_fname_pred3, "rb")))
    # next_list_of_test_dict.append(pickle.load(open(dict_fname_pred4, "rb")))
    # next_list_of_test_dict.append(pickle.load(open(dict_fname_pred5, "rb")))
    # print(pred2, pred3, pred4, pred5, "this is a test")
    return [pred2, pred3, pred4, pred5]

def exist_or_no(string):
    if string == '' or string == False:
        return 0
    else:
        return 1

def none_or_str(string):
    if string == '':
        return None
    else:
        return string


def cleanInstruction(instruction):
    instruction = instruction.lower()
    instruction = ''.join(ch for ch in instruction if ch not in exclude)
    return instruction


def get_arguments_test(test_dict, traj_data):
    """
    Retrieves and processes arguments from the provided test data and test dictionary.

    Parameters:
    - test_dict: A dictionary containing task-specific information.
    - traj_data: Data related to the trajectory and task.

    Returns:
    - instruction: A processed instruction string.
    - task_type: The type of the task.
    - mrecep_target: The target movable receptacle.
    - object_target: The target object to manipulate.
    - parent_target: The target parent receptacle or location.
    - sliced: A flag indicating if the object is sliced (1 for sliced, 0 for unsliced).
    """
    r_idx = traj_data['repeat_idx']
    high_level_lang = traj_data['turk_annotations']['anns'][r_idx]['task_desc']
    low_level_lang = traj_data['turk_annotations']['anns'][r_idx]['high_descs']

    instructions = [high_level_lang] + low_level_lang
    # print(instructions, "this is a test")
    instructions = [cleanInstruction(instruction) for instruction in instructions] # changes all instructions to lowercase and removes punctuation
    instruction = '[SEP]'.join(instructions)

    # print(test_dict, "this is a test")
    task_type, mrecep_target, object_target, parent_target, sliced = \
        test_dict[instruction]['task_type'], \
            test_dict[instruction]['mrecep_target'], \
                test_dict[instruction]['object_target'], \
                    test_dict[instruction]['parent_target'],\
             test_dict[instruction]['sliced']
    
    # print(object_target, "this is a test")
    # mrecep_target = 'Mug'
    if isinstance(task_type, int):
        task_type = task_type_dict[task_type]
    return instruction, task_type, mrecep_target, object_target, parent_target, sliced 


def get_arguments(traj_data):
    task_type = traj_data['task_type']
    try:
        r_idx = traj_data['repeat_idx']
    except:
        r_idx = 0
    language_goal_instr = traj_data['turk_annotations']['anns'][r_idx]['task_desc']
    
    sliced = exist_or_no(traj_data['pddl_params']['object_sliced'])
    mrecep_target = none_or_str(traj_data['pddl_params']['mrecep_target'])
    object_target = none_or_str(traj_data['pddl_params']['object_target'])
    parent_target = none_or_str(traj_data['pddl_params']['parent_target'])
    #toggle_target = none_or_str(traj_data['pddl_params']['toggle_target'])
    
    return language_goal_instr, task_type, mrecep_target, object_target, parent_target, sliced


def add_target(target, target_action, list_of_actions):
    if target in [a for a in constants.OPENABLE_CLASS_LIST if not(a == 'Box')]:
        list_of_actions.append((target, "OpenObject"))
    list_of_actions.append((target, target_action))
    if target in [a for a in constants.OPENABLE_CLASS_LIST if not(a == 'Box')]:
        list_of_actions.append((target, "CloseObject"))
    return list_of_actions

def determine_consecutive_interx(list_of_actions, previous_pointer, sliced=False):
    returned, target_instance = False, None
    if previous_pointer <= len(list_of_actions)-1:
        if list_of_actions[previous_pointer][0] == list_of_actions[previous_pointer+1][0]:
            returned = True
            #target_instance = list_of_target_instance[-1] #previous target
            target_instance = list_of_actions[previous_pointer][0]
        #Micorwave or Fridge
        elif list_of_actions[previous_pointer][1] == "OpenObject" and list_of_actions[previous_pointer+1][1] == "PickupObject":
            returned = True
            #target_instance = list_of_target_instance[0]
            target_instance = list_of_actions[0][0]
            if sliced:
                #target_instance = list_of_target_instance[3]
                target_instance = list_of_actions[3][0]
        #Micorwave or Fridge
        elif list_of_actions[previous_pointer][1] == "PickupObject" and list_of_actions[previous_pointer+1][1] == "CloseObject":
            returned = True
            #target_instance = list_of_target_instance[-2] #e.g. Fridge
            target_instance = list_of_actions[previous_pointer-1][0]
        #Faucet
        elif list_of_actions[previous_pointer+1][0] == "Faucet" and list_of_actions[previous_pointer+1][1] in ["ToggleObjectOn", "ToggleObjectOff"]:
            returned = True
            target_instance = "Faucet"
        #Pick up after faucet 
        elif list_of_actions[previous_pointer][0] == "Faucet" and list_of_actions[previous_pointer+1][1] == "PickupObject":
            returned = True
            #target_instance = list_of_target_instance[0]
            target_instance = list_of_actions[0][0]
            if sliced:
                #target_instance = list_of_target_instance[3]
                target_instance = list_of_actions[3][0]
    return returned, target_instance

def get_next_list_of_actions(original_list_of_actions, new_obj, old_obj):

        modified_list = [(new_obj if item[0] == old_obj else item[0], item[1]) for item in original_list_of_actions]
        return original_list_of_actions + modified_list
def get_list_of_highlevel_actions(traj_data, test_dict=None, args_nonsliced=False):
    # print(test_dict, "just testing")
    language_goal, task_type, mrecep_target, obj_target, parent_target, sliced = get_arguments_test(test_dict, traj_data)

    #obj_target = 'Tomato'
    # mrecep_target = "Mug"
    if parent_target == "Sink":
        parent_target = "SinkBasin"
    if parent_target == "Bathtub":
        parent_target = "BathtubBasin"
    
    #Change to this after the sliced happens
    if args_nonsliced:
        if sliced == 1:
            obj_target = obj_target +'Sliced'
        #Map sliced as the same place in the map, but like "|SinkBasin" look at the objectid

    
    categories_in_inst = []
    list_of_highlevel_actions = []
    second_object = []
    caution_pointers = []
    #obj_target = "Tomato"

    if sliced == 1:
       list_of_highlevel_actions.append(("Knife", "PickupObject"))
       list_of_highlevel_actions.append((obj_target, "SliceObject"))
       caution_pointers.append(len(list_of_highlevel_actions))
       list_of_highlevel_actions.append(("SinkBasin", "PutObject"))
       categories_in_inst.append(obj_target)
       
    if sliced:
        obj_target = obj_target +'Sliced'

    
    if task_type == 'pick_cool_then_place_in_recep': #0 in new_labels 
       list_of_highlevel_actions.append((obj_target, "PickupObject"))
       caution_pointers.append(len(list_of_highlevel_actions))
       list_of_highlevel_actions = add_target("Fridge", "PutObject", list_of_highlevel_actions)
       list_of_highlevel_actions.append(("Fridge", "OpenObject"))
       list_of_highlevel_actions.append((obj_target, "PickupObject"))
       list_of_highlevel_actions.append(("Fridge", "CloseObject"))
       caution_pointers.append(len(list_of_highlevel_actions))
       list_of_highlevel_actions = add_target(parent_target, "PutObject", list_of_highlevel_actions)
       categories_in_inst.append(obj_target)
       categories_in_inst.append("Fridge")
       categories_in_inst.append(parent_target)
       
    elif task_type == 'pick_and_place_with_movable_recep': #1 in new_labels
        list_of_highlevel_actions.append((obj_target, "PickupObject"))
        caution_pointers.append(len(list_of_highlevel_actions))
        list_of_highlevel_actions = add_target(mrecep_target, "PutObject", list_of_highlevel_actions)
        list_of_highlevel_actions.append((mrecep_target, "PickupObject"))
        caution_pointers.append(len(list_of_highlevel_actions))
        list_of_highlevel_actions = add_target(parent_target, "PutObject", list_of_highlevel_actions)
        categories_in_inst.append(obj_target)
        categories_in_inst.append(mrecep_target)
        categories_in_inst.append(parent_target)

    
    elif task_type == 'pick_and_place_simple':#2 in new_labels 
        list_of_highlevel_actions.append((obj_target, "PickupObject"))
        caution_pointers.append(len(list_of_highlevel_actions))
        list_of_highlevel_actions = add_target(parent_target, "PutObject", list_of_highlevel_actions)
        #list_of_highlevel_actions.append((parent_target, "PutObject"))
        categories_in_inst.append(obj_target)
        categories_in_inst.append(parent_target)
        
    
    elif task_type == 'pick_heat_then_place_in_recep': #4 in new_labels
        list_of_highlevel_actions.append((obj_target, "PickupObject"))  
        caution_pointers.append(len(list_of_highlevel_actions))
        list_of_highlevel_actions = add_target("Microwave", "PutObject", list_of_highlevel_actions)
        list_of_highlevel_actions.append(("Microwave", "ToggleObjectOn" ))
        list_of_highlevel_actions.append(("Microwave", "ToggleObjectOff" ))
        list_of_highlevel_actions.append(("Microwave", "OpenObject"))
        list_of_highlevel_actions.append((obj_target, "PickupObject"))
        list_of_highlevel_actions.append(("Microwave", "CloseObject"))
        caution_pointers.append(len(list_of_highlevel_actions))
        list_of_highlevel_actions = add_target(parent_target, "PutObject", list_of_highlevel_actions)
        categories_in_inst.append(obj_target)
        categories_in_inst.append("Microwave")
        categories_in_inst.append(parent_target)
        
    elif task_type == 'pick_two_obj_and_place': #3 in new_labels
        list_of_highlevel_actions.append((obj_target, "PickupObject"))  
        caution_pointers.append(len(list_of_highlevel_actions))
        list_of_highlevel_actions = add_target(parent_target, "PutObject", list_of_highlevel_actions)
        if parent_target in constants.OPENABLE_CLASS_LIST:
            second_object = [False] * 4
        else:
            second_object = [False] * 2
        if sliced:
            second_object = second_object + [False] * 3
        caution_pointers.append(len(list_of_highlevel_actions))
        list_of_highlevel_actions.append((obj_target, "PickupObject"))  
        #caution_pointers.append(len(list_of_highlevel_actions))
        second_object.append(True)
        caution_pointers.append(len(list_of_highlevel_actions))
        list_of_highlevel_actions = add_target(parent_target, "PutObject", list_of_highlevel_actions)
        second_object.append(False)
        categories_in_inst.append(obj_target)
        categories_in_inst.append(parent_target)
        
        
    elif task_type == 'look_at_obj_in_light': #5 in new_labels
        list_of_highlevel_actions.append((obj_target, "PickupObject"))
        #if toggle_target == "DeskLamp":
        #    print("Original toggle target was DeskLamp")
        toggle_target = "FloorLamp"
        list_of_highlevel_actions.append((toggle_target, "ToggleObjectOn" ))
        categories_in_inst.append(obj_target)
        categories_in_inst.append(toggle_target)
        
    elif task_type == 'pick_clean_then_place_in_recep': #6 in new_labels
        list_of_highlevel_actions.append((obj_target, "PickupObject"))  
        caution_pointers.append(len(list_of_highlevel_actions))
        list_of_highlevel_actions.append(("SinkBasin", "PutObject"))  #Sink or SinkBasin? 
        list_of_highlevel_actions.append(("Faucet", "ToggleObjectOn"))
        list_of_highlevel_actions.append(("Faucet", "ToggleObjectOff"))
        list_of_highlevel_actions.append((obj_target, "PickupObject"))  
        caution_pointers.append(len(list_of_highlevel_actions))
        list_of_highlevel_actions = add_target(parent_target, "PutObject", list_of_highlevel_actions)
        categories_in_inst.append(obj_target)
        categories_in_inst.append("SinkBasin")
        categories_in_inst.append("Faucet")
        categories_in_inst.append(parent_target)
    else:
        raise Exception("Task type not one of 0, 1, 2, 3, 4, 5, 6!")

    if sliced == 1:
       if not(parent_target == "SinkBasin"):
            categories_in_inst.append("SinkBasin")
    
    #return [(goal_category, interaction), (goal_category, interaction), ...]
    # print("instruction goal is to", language_goal)
    #list_of_highlevel_actions = [ ('Microwave', 'OpenObject'), ('Microwave', 'PutObject'), ('Microwave', 'CloseObject')]
    #list_of_highlevel_actions = [('Microwave', 'OpenObject'), ('Microwave', 'PutObject'), ('Microwave', 'CloseObject'), ('Microwave', 'ToggleObjectOn'), ('Microwave', 'ToggleObjectOff'), ('Microwave', 'OpenObject'), ('Apple', 'PickupObject'), ('Microwave', 'CloseObject'), ('Fridge', 'OpenObject'), ('Fridge', 'PutObject'), ('Fridge', 'CloseObject')]
    #categories_in_inst = ['Microwave', 'Fridge']
    return language_goal, list_of_highlevel_actions, categories_in_inst, second_object, caution_pointers
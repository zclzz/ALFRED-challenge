from curses.ascii import ctrl
import pickle
from models.semantic_policy.sem_map_model import UNetMulti, MLM
import alfred_utils.gen.constants as constants
from models.instructions_processed_LP.ALFRED_task_helper import determine_consecutive_interx
from models.sem_mapping import Semantic_Mapping
import envs.utils.pose as pu
from envs import make_vec_envs
from arguments import get_args
from datetime import datetime
from collections import defaultdict
import skimage.morphology
import math
import numpy as np
import torch.nn as nn
import torch
import cv2
import os
import sys
import matplotlib
# from models.segmentation.segmentation_helper import SemgnetationHelper

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ["OMP_NUM_THREADS"] = "1"

if sys.platform == 'darwin':
    matplotlib.use("tkagg")


def into_grid(ori_grid, grid_size):
    """
    Converts an input grid to a grid of a specified size using averaging.

    Parameters:
    - ori_grid: The original grid to be converted.
    - grid_size: The desired size of the output grid.

    Returns:
    - return_grid: The converted grid of the specified size.
    """
    if ori_grid.shape[0] == grid_size:
        return ori_grid

    one_cell_size = math.ceil(240 / grid_size)

    ori_grid = ori_grid.unsqueeze(0).unsqueeze(0)

    m = nn.AvgPool2d(one_cell_size, stride=one_cell_size)
    avg_pooled = m(ori_grid)[0, 0, :, :]
    return_grid = (avg_pooled > 0).float()

    return return_grid


def getCurrImgCoord(planner_pose_input, map_resolution):
    """
    Gets the current image coordinates based on planner pose input.

    Parameters:
    - planner_pose_input: Input information about planner pose.
    - map_resolution: The resolution of the map.

    Returns:
    - yx: The current image coordinates.
    """
    start_x, start_y, start_o, gx1, gx2, gy1, gy2 = planner_pose_input
    gx1, gx2, gy1, gy2 = int(gx1), int(gx2), int(gy1), int(gy2)
    r, c = start_y, start_x
    yx = [int(r * 100.0/map_resolution - gy1),
          int(c * 100.0/map_resolution - gx1)]
    return yx


def getNeighborMap(ori_map, dil_sz):
    """
    Generates a neighboring map by dilating the original map.

    Parameters:
    - ori_map: The original map.
    - dil_sz: The size of dilation.

    Returns:
    - goal_neighbor: The generated neighboring map.
    """
    goal_neighbor = ori_map.copy()
    goal_neighbor = skimage.morphology.binary_dilation(
        goal_neighbor, skimage.morphology.square(dil_sz))
    return goal_neighbor


def searchArgmax(conv_sz, score_map, mask=None):
    """
    Searches for the maximum value in a score map within a specified convolution size.

    Parameters:
    - conv_sz: The convolution size.
    - score_map: The score map to search.
    - mask: An optional mask for the score map.

    Returns:
    - max_coords: The coordinates of the maximum value in the score map.
    """
    # fast 2D convolution
    kernel = np.ones(conv_sz)
    conv_1d = lambda m: np.convolve(m, kernel, mode='same')
    ver_sum = np.apply_along_axis(conv_1d, axis=0, arr=score_map)
    conved = np.apply_along_axis(conv_1d, axis=1, arr=ver_sum)

    conved_masked = conved if (mask is None) else conved * mask

    return np.unravel_index(conved_masked.argmax(), conved_masked.shape)


def main():
    args = get_args() # stores a bunch of arguments in a dictionary
    # print('args.splits', args.splits) #args.splits alfred_data_small/splits/oct21.json
    # print('args.eval_split', args.eval_split) # args.eval_split valid_unseen
    # print('args.from_idx', args.from_idx) # args.from_idx 203

    # print(args)
    dn = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    args.dn = dn
    if args.set_dn != "":
        args.dn = args.set_dn
        dn = args.set_dn
    print("dn is ", dn) # dn stands

    # Create results folder
    if not os.path.exists("results/logs"):
        os.makedirs("results/logs")
    if not os.path.exists("results/leaderboard"):
        os.makedirs("results/leaderboard")
    if not os.path.exists("results/successes"):
        os.makedirs("results/successes")
    if not os.path.exists("results/fails"):
        os.makedirs("results/fails")
    if not os.path.exists("results/analyze_recs"):
        os.makedirs("results/analyze_recs")

    completed_episodes = []
    seen_objects = {}
    skip_indices = {}
    if args.exclude_list != "":
        if args.exclude_list[-2:] == ".p":
            skip_indices = pickle.load(open(args.exclude_list, 'rb'))
            skip_indices = {int(s): 1 for s in skip_indices}
        else:
            skip_indices = [a for a in args.exclude_list.split(',')]
            skip_indices = {int(s): 1 for s in skip_indices}
    args.skip_indices = skip_indices
    actseqs = []
    all_completed = [False] * args.num_processes
    # print('all completed:', all_completed)  
    successes = []
    failures = []
    analyze_recs = []
    traj_number = [0] * args.num_processes
    num_scenes = args.num_processes

    # Set seeds
    local_rngs = [np.random.RandomState(args.seed + args.from_idx + e) for e in range(args.num_processes)]
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # Set up logging
    large_objects2idx = {obj: i for i, obj in enumerate(
        constants.map_save_large_objects)}
    all_objects2idx = {o: i for i, o in enumerate(constants.map_all_objects)}
    softmax = nn.Softmax(dim=1)

    # Logging and loss variables
    num_episodes = [0] * args.num_processes
    for e in range(args.from_idx, args.to_idx):
        remainder = e % args.num_processes
        num_episodes[remainder] += 1

    # Set up device
    device = args.device = torch.device(
        "cuda:" + args.which_gpu if args.cuda else "cpu")
    if args.sem_policy_type == "mlm":
        Unet_model = MLM(
            (240, 240), (args.grid_sz, args.grid_sz), f"models/semantic_policy/{args.mlm_fname}.csv",
            options=args.mlm_options).to(device=device)
        if "mixed_search" in args.mlm_options:
            Unet_model_equal = MLM(
                (240, 240), (args.grid_sz, args.grid_sz),
                f"models/semantic_policy/mlmscore_equal.csv",
                options=args.mlm_options).to(device=device)

    elif args.sem_policy_type == "cnn":
        assert args.grid_sz == 8, "grid size should be 8 when sem_policy_type is 'film'"
        Unet_model = UNetMulti(
            (240, 240), num_sem_categories=24).to(device=device)
        sd = torch.load(
            'models/semantic_policy/new_best_model.pt', map_location=device)
        Unet_model.load_state_dict(sd)
        del sd

    finished = np.zeros((args.num_processes))

    # Starting environments
    torch.set_num_threads(1)
    envs = make_vec_envs(args)
    fails = [0] * num_scenes
    prev_cns = [None] * num_scenes

    obs, infos, actions_dicts = envs.load_initial_scene()
    # actions_dicts[0]['list_of_actions'] = [('Mug', 'PickupObject'), ('Cabinet', 'OpenObject')]
    # print('actions_dicts:', actions_dicts)
    # goal_instr = actions_dicts[0]['goal_instr']
    # list_of_list_of_actions = actions_dicts[0]['list_of_list_of_actions']
    explore_steps = 100
    num_action_seq = 1
    done_seq = None
    need_explore = False if num_action_seq == 1 else True
    started = False
    receptacles = {'ArmChair', 'BathtubBasin', 'Bed', 'CoffeeTable', 
               'CounterTop', 'Desk', 'DiningTable', 'Dresser', 'GarbageCan', 
               'Shelf', 'SideTable', 'SinkBasin', 'Sofa', 'StoveBurner', 'TVStand', 'Toilet'}

    most_prominent_receptacle = None
    second_objects = []
    list_of_actions_s = []
    list_of_list_of_actions_s = []
    task_types = []
    whether_sliced_s = []


    for e in range(args.num_processes):
        # print('actions_dicts[e]:', actions_dicts[e])
        second_object = actions_dicts[e]['second_object']
        # actions_dicts[e]['list_of_actions'] = [('CellPhone', 'PickupObject'), ('FloorLamp', 'ToggleObjectOn')]
        list_of_actions = actions_dicts[e]['list_of_actions']
        list_of_list_of_actions = actions_dicts[e]['list_of_list_of_actions']
        # print('list actions:', list_of_actions)
        # print('actions dict:', actions_dicts)
        goal_instr = actions_dicts[e]['goal_instr']

        task_type = actions_dicts[e]['task_type']
        sliced = actions_dicts[e]['sliced']
        second_objects.append(second_object)
        list_of_actions_s.append(list_of_actions)
        list_of_list_of_actions_s.append(list_of_list_of_actions)
        task_types.append(task_type)
        whether_sliced_s.append(sliced)

    # print('list of actions_s:', list_of_actions_s)
    task_finish = [False] * args.num_processes
    first_steps = [True] * args.num_processes
    num_steps_so_far = [0] * args.num_processes
    subgoal_steps = [0] * args.num_processes
    load_goal_pointers = [0] * args.num_processes
    list_of_actions_pointer_s = [0] * args.num_processes
    # print('list_of_actions_pointer_s:', list_of_actions_pointer_s)
    goal_spotted_s = [False] * args.num_processes
    list_of_actions_pointer_s = [0] * args.num_processes
    # print('list of actions:', list_of_actions_pointer_s)
    goal_logs = [[] for i in range(args.num_processes)]
    goal_cat_before_second_objects = [None] * args.num_processes
    subgoal_counter_s = [0] * args.num_processes
    found_subgoal_coordinates = [None] * args.num_processes

    do_not_update_cat_s = [None] * args.num_processes
    wheres_delete_s = [np.zeros((240, 240))] * args.num_processes
    sem_search_searched_s = [np.zeros((240, 240))] * args.num_processes

    args.num_sem_categories = 1 + 1 + 1 + 5 * args.num_processes
    if args.sem_policy_type != "none":
        args.num_sem_categories = args.num_sem_categories + 23
    obs = torch.tensor(obs).to(device)


    torch.set_grad_enabled(False)

    # Initialize map variables
    # Full map consists of multiple channels containing the following:
    # 1. Obstacle Map
    # 2. Exploread Area
    # 3. Current Agent Location
    # 4. Past Agent Locations
    # 5-: Semantic categories, as defined in sem_exp_thor.total_cat2idx
    # i.e. 'Knife': 0, 'SinkBasin': 1, 'ArmChair': 2, 'BathtubBasin': 3, 'Bed': 4, 'Cabinet': 5, 'Cart': 6, 'CoffeeMachine': 7, 'CoffeeTable': 8, 'CounterTop': 9, 'Desk': 10, 'DiningTable': 11, 'Drawer': 12, 'Dresser': 13, 'Fridge': 14, 'GarbageCan': 15, 'Microwave': 16, 'Ottoman': 17, 'Safe': 18, 'Shelf': 19, 'SideTable': 20, 'Sofa': 21, 'StoveBurner': 22, 'TVStand': 23, 'Toilet': 24, 'CellPhone': 25, 'FloorLamp': 26, 'None': 29
    nc = args.num_sem_categories + 4  # num channels

    # Calculating full and local map sizes
    map_size = args.map_size_cm // args.map_resolution
    full_w, full_h = map_size, map_size
    local_w, local_h = int(full_w / args.global_downscaling), \
        int(full_h / args.global_downscaling)

    full_map = torch.zeros(num_scenes, nc, full_w, full_h).float().to(device)
    local_map = torch.zeros(num_scenes, nc, local_w,
                            local_h).float().to(device)

    # Initial full and local pose
    full_pose = torch.zeros(num_scenes, 3).float().to(device)
    local_pose = torch.zeros(num_scenes, 3).float().to(device)

    # Origin of local map
    origins = np.zeros((num_scenes, 3))

    # Local Map Boundaries
    lmb = np.zeros((num_scenes, 4)).astype(int)

    # Planner pose inputs has 7 dimensions
    # 1-3 store continuous global agent location
    # 4-7 store local map boundaries
    planner_pose_inputs = np.zeros((num_scenes, 7))

    def get_local_map_boundaries(agent_loc, local_sizes, full_sizes):
        loc_r, loc_c = agent_loc
        local_w, local_h = local_sizes
        full_w, full_h = full_sizes

        if args.global_downscaling > 1:
            gx1, gy1 = loc_r - local_w // 2, loc_c - local_h // 2
            gx2, gy2 = gx1 + local_w, gy1 + local_h
            if gx1 < 0:
                gx1, gx2 = 0, local_w
            if gx2 > full_w:
                gx1, gx2 = full_w - local_w, full_w

            if gy1 < 0:
                gy1, gy2 = 0, local_h
            if gy2 > full_h:
                gy1, gy2 = full_h - local_h, full_h
        else:
            gx1, gx2, gy1, gy2 = 0, full_w, 0, full_h

        return [gx1, gx2, gy1, gy2]

    def init_map_and_pose():
        full_map.fill_(0.)
        full_pose.fill_(0.)
        full_pose[:, :2] = args.map_size_cm / 100.0 / 2.0

        locs = full_pose.cpu().numpy()
        planner_pose_inputs[:, :3] = locs
        for e in range(num_scenes):
            r, c = locs[e, 1], locs[e, 0]
            loc_r, loc_c = [int(r * 100.0 / args.map_resolution),
                            int(c * 100.0 / args.map_resolution)]

            full_map[e, 2:4, loc_r - 1:loc_r + 2, loc_c - 1:loc_c + 2] = 1.0

            lmb[e] = get_local_map_boundaries((loc_r, loc_c),
                                              (local_w, local_h),
                                              (full_w, full_h))

            planner_pose_inputs[e, 3:] = lmb[e]
            origins[e] = [lmb[e][2] * args.map_resolution / 100.0,
                          lmb[e][0] * args.map_resolution / 100.0, 0.]

        for e in range(num_scenes):
            local_map[e] = full_map[e, :, lmb[e, 0]
                :lmb[e, 1], lmb[e, 2]:lmb[e, 3]]
            local_pose[e] = full_pose[e] - \
                torch.from_numpy(origins[e]).to(device).float()

    def init_map_and_pose_for_env(e):
        full_map[e].fill_(0.)
        full_pose[e].fill_(0.)
        full_pose[e, :2] = args.map_size_cm / 100.0 / 2.0

        locs = full_pose[e].cpu().numpy()
        planner_pose_inputs[e, :3] = locs
        r, c = locs[1], locs[0]
        loc_r, loc_c = [int(r * 100.0 / args.map_resolution),
                        int(c * 100.0 / args.map_resolution)]

        full_map[e, 2:4, loc_r - 1:loc_r + 2, loc_c - 1:loc_c + 2] = 1.0

        lmb[e] = get_local_map_boundaries(
            (loc_r, loc_c), (local_w, local_h), (full_w, full_h))

        planner_pose_inputs[e, 3:] = lmb[e]
        origins[e] = [lmb[e][2] * args.map_resolution / 100.0,
                      lmb[e][0] * args.map_resolution / 100.0, 0.]

        local_map[e] = full_map[e, :, lmb[e, 0]:lmb[e, 1], lmb[e, 2]:lmb[e, 3]]
        local_pose[e] = full_pose[e] - \
            torch.from_numpy(origins[e]).to(device).float()
        
    def get_reordered_actionSeq(list_of_list_of_actions, seen_objects, task_type, language_goal):
        receptacles = {'ArmChair', 'BathtubBasin', 'Bed', 'Cabinet', 'Cart', 'CoffeeMachine', 'CoffeeTable', 
                    'CounterTop', 'Desk', 'DiningTable', 'Drawer', 'Dresser', 'Fridge', 'GarbageCan', 
                    'Microwave', 'Ottoman', 'Safe', 'Shelf', 'SideTable', 'SinkBasin', 'Sofa', 
                    'StoveBurner', 'TVStand', 'Toilet'}

        # Normalize the seen objects score to prevent the counter from going very high
        max_seen = max(seen_objects.values())
        normalized_scores = {obj: score/max_seen for obj, score in seen_objects.items()}
        # normalized_scores['None'] = 0
        # Aggregate scores for each sequence of actions based on normalized scores of seen objects
        seq_scores = []
        for index, seq in enumerate(list_of_list_of_actions):
            score = 0
            all_seen = True  # Assume initially that all objects in the sequence are seen
            for action in seq:
                obj = action[0]
                if task_type == 'pick_cool_then_place_in_recep' and obj == 'Fridge':
                    score += 5
                elif task_type == 'pick_heat_then_place_in_recep' and obj == 'Microwave':
                    score += 5
                elif task_type == 'pick_clean_then_place_in_recep' and obj == 'Faucet':
                    score += 5
                elif task_type == 'look_at_obj_in_light' and 'Lamp' in obj:
                    score += 5
                elif 'slice' in language_goal or 'cut' in language_goal or 'piece' in language_goal:
                    if obj == 'Knife' or obj == 'ButterKnife':
                        score += 5

                # Merge 'Sink' and 'SinkBasin', as well as 'Bathtub' and 'BathtubBasin'
                if obj == 'SinkBasin':
                    obj = 'Sink'
                elif obj == 'BathtubBasin':
                    obj = 'Bathtub'
                elif obj == 'ButterKnife' or obj == 'Knife':
                    obj = 'Knife'
                # Calculate score, penalize if object not seen
                if obj in normalized_scores: # Add normalized scores of seen objects
                    score += normalized_scores[obj]
                else:  # Object not seen
                    if obj == None or not(obj.endswith('Sliced')) or (obj.endswith('Sliced') and obj[:-6] not in normalized_scores):  # Check if the object is a sliced version of a seen object
                        if (obj != 'Microwave'):
                            all_seen = False
                        if obj in receptacles and obj != 'Microwave': # Penalize unseen receptacles more
                            score -= 15
                        elif obj != 'Microwave': # Penalize unseen except microwave
                            score -= 0.5
            # If it's the first sequence and all objects are seen, add an extra 25 to the score
            if index == 0 and all_seen:
                score += 25
            seq_scores.append((score, seq))

        # Sort sequences based on their scores, highest first
        seq_scores.sort(reverse=True, key=lambda x: x[0])
        reordered_seqs = [seq for _, seq in seq_scores]

        return reordered_seqs

    init_map_and_pose()

    # slam
    sem_map_module = Semantic_Mapping(args).to(device)
    sem_map_module.eval()
    sem_map_module.set_view_angles([45] * args.num_processes)

    # Predict semantic map from frame 1
    poses = torch.from_numpy(np.asarray(
        [infos[env_idx]['sensor_pose'] for env_idx in range(num_scenes)])
    ).float().to(device)

    _, local_map, _, local_pose = \
        sem_map_module(obs, poses, local_map, local_pose)

    # Compute Global policy input
    locs = local_pose.cpu().numpy()

    for e in range(num_scenes):
        r, c = locs[e, 1], locs[e, 0]
        loc_r, loc_c = [int(r * 100.0 / args.map_resolution),
                        int(c * 100.0 / args.map_resolution)]

        local_map[e, 2:4, loc_r - 1:loc_r + 2, loc_c - 1:loc_c + 2] = 1.

    # For now
    global_goals = []
    for e in range(num_scenes):
        c1 = local_rngs[e].choice(local_w)
        c2 = local_rngs[e].choice(local_h)
        global_goals.append((c1, c2))

    goal_maps = [np.zeros((local_w, local_h)) for _ in range(num_scenes)]

    for e in range(num_scenes):
        goal_maps[e][global_goals[e][0], global_goals[e][1]] = 1


    planner_inputs = [{} for e in range(num_scenes)]
    for e, p_input in enumerate(planner_inputs):
        p_input['map_pred'] = local_map[e, 0, :, :].cpu().numpy()
        p_input['exp_pred'] = local_map[e, 1, :, :].cpu().numpy()
        p_input['pose_pred'] = planner_pose_inputs[e]
        p_input['goal'] = goal_maps[e]
        p_input['found_goal'] = 0
        p_input['wait'] = finished[e]
        p_input['list_of_actions'] = list_of_actions_s[e]
        p_input['list_of_list_of_actions'] = list_of_list_of_actions_s[e]
        p_input['list_of_actions_pointer'] = list_of_actions_pointer_s[e]
        p_input['consecutive_interaction'] = None
        p_input['consecutive_target'] = None
        p_input['manual_step'] = None
        if args.visualize or args.print_images:
            local_map[e, -1, :, :] = 1e-5
            p_input['sem_map_pred'] = local_map[e, 4:, :,
                                                :].argmax(0).cpu().numpy()
    obs, rew, done, infos, goal_success_s, next_step_dict_s = envs.plan_act_and_preprocess(
        planner_inputs, goal_spotted_s)

    goal_success_s = list(goal_success_s)
    view_angles = []
    for e in range(num_scenes):
        next_step_dict = next_step_dict_s[e]
        view_angle = next_step_dict['view_angle']
        view_angles.append(view_angle)

        fails[e] += next_step_dict['fails_cur']

    sem_map_module.set_view_angles(view_angles)

    failed_msg_printed = False

    consecutive_interaction_s, target_instance_s = [None]*num_scenes, [None]*num_scenes
    for e in range(num_scenes):
        num_steps_so_far[e] = next_step_dict_s[e]['steps_taken']
        first_steps[e] = False
        if goal_success_s[e]: # after completion of a subgoal
            print('subgoal_completed')
            if list_of_actions_pointer_s[e] == len(list_of_actions_s[e]) - 1:
                all_completed[e] = True
            else:
                subgoal_counter_s[e] = 0
                found_subgoal_coordinates[e] = None
                list_of_actions_pointer_s[e] += 1
                goal_name = list_of_actions_s[e][list_of_actions_pointer_s[e]][0]
                reset_goal_true_false = [False] * num_scenes
                reset_goal_true_false[e] = True

                # If consecutive interactions,
                returned, target_instance_s[e] = determine_consecutive_interx(
                    list_of_actions_s[e], list_of_actions_pointer_s[e]-1, whether_sliced_s[e])
                if returned:
                    consecutive_interaction_s[e] = list_of_actions_s[e][list_of_actions_pointer_s[e]][1]

                infos = envs.reset_goal(
                    reset_goal_true_false, goal_name, consecutive_interaction_s)

    torch.set_grad_enabled(False)
    spl_per_category = defaultdict(list)
    success_per_category = defaultdict(list)

    for _ in range(args.num_training_frames//args.num_processes):
        skip_save_pic = task_finish.copy()
        # Reinitialize variables when episode ends
        for e, x in enumerate(task_finish):
            if x:
                spl = infos[e]['spl']
                success = infos[e]['success']
                dist = infos[e]['distance_to_goal']
                spl_per_category[infos[e]['goal_name']].append(spl)
                success_per_category[infos[e]['goal_name']].append(success)
                traj_number[e] += 1
                init_map_and_pose_for_env(e)

                if not(finished[e]):
                    # load next episode for env
                    number_of_this_episode = args.from_idx + \
                        traj_number[e] * num_scenes + e
                    print("steps taken for episode# ",  number_of_this_episode -
                          num_scenes, " is ", next_step_dict_s[e]['steps_taken'])
                    completed_episodes.append(number_of_this_episode)
                    pickle.dump(
                        completed_episodes,
                        open(f"results/completed_episodes_{args.eval_split}{args.from_idx}_to_{args.to_idx}_{dn}.p", 'wb'))
                    if args.leaderboard and args.test:
                        if args.test_seen:
                            add_str = "seen"
                        else:
                            add_str = "unseen"
                        pickle.dump(actseqs, open(
                            f"results/leaderboard/actseqs_test_{add_str}_{dn}_{args.from_idx}_to_{args.to_idx}.p", "wb"))
                    load = [False] * args.num_processes
                    load[e] = True
                    do_not_update_cat_s[e] = None
                    wheres_delete_s[e] = np.zeros((240, 240))
                    sem_search_searched_s[e] = np.zeros((240, 240))
                    obs, infos, actions_dicts = envs.load_next_scene(load)
                    local_rngs[e] = np.random.RandomState(args.seed + number_of_this_episode)
                    view_angles[e] = 45
                    sem_map_module.set_view_angles(view_angles)
                    if actions_dicts[e] is None:
                        finished[e] = True
                    else:
                        second_objects[e] = actions_dicts[e]['second_object']
                        # print("second object is ", second_objects[e])
                        list_of_actions_s[e] = actions_dicts[e]['list_of_actions']
                        list_of_list_of_actions_s[e] = actions_dicts[e]['list_of_list_of_actions']

                        task_types[e] = actions_dicts[e]['task_type']
                        whether_sliced_s[e] = actions_dicts[e]['sliced']

                        task_finish[e] = False
                        num_steps_so_far[e] = 0
                        list_of_actions_pointer_s[e] = 0
                        goal_spotted_s[e] = False
                        found_goal[e] = 0
                        subgoal_counter_s[e] = 0
                        found_subgoal_coordinates[e] = None
                        first_steps[e] = True

                        all_completed[e] = False
                        goal_success_s[e] = False

                        obs = torch.tensor(obs).to(device)
                        fails[e] = 0
                        goal_logs[e] = []
                        goal_cat_before_second_objects[e] = None

        if sum(finished) == args.num_processes:
            print("all finished")
            if args.leaderboard and args.test:
                if args.test_seen:
                    add_str = "seen"
                else:
                    add_str = "unseen"
                pickle.dump(actseqs, open(
                    "results/leaderboard/actseqs_test_" + add_str + "_" + dn + ".p", "wb"))
            break

        # ------------------------------------------------------------------
        # Semantic Mapping Module
        poses = torch.from_numpy(np.asarray(
            [infos[env_idx]['sensor_pose'] for env_idx
             in range(num_scenes)])
        ).float().to(device)

        _, local_map, _, local_pose = sem_map_module(
            obs, poses, local_map, local_pose, build_maps=True, no_update=False)

        locs = local_pose.cpu().numpy()
        planner_pose_inputs[:, :3] = locs + origins
        local_map[:, 2, :, :].fill_(0.)  # Resetting current location channel
        for e in range(num_scenes):
            r, c = locs[e, 1], locs[e, 0]
            loc_r, loc_c = [int(r * 100.0 / args.map_resolution),
                            int(c * 100.0 / args.map_resolution)]
            local_map[e, 2:4, loc_r - 2:loc_r + 3, loc_c - 2:loc_c + 3] = 1.

        for e in range(num_scenes):
            new_goal_needed = args.delete_from_map_after_move_until_visible and (next_step_dict_s[e]['next_goal'] or next_step_dict_s[e]['delete_lamp'])
            if new_goal_needed:
                ep_num = args.from_idx + traj_number[e] * num_scenes + e

                # search failed, so delete regions neigboring the previous goal
                # TODO: use disk dilation and connected entity for this part?
                goal_neighbor_gl = getNeighborMap(goal_maps[e], dil_sz=args.goal_search_del_size)
                wheres_delete_s[e][np.where(goal_neighbor_gl == 1)] = 1

                goal_neighbor_ss = getNeighborMap(goal_maps[e], dil_sz=args.sem_search_del_size)
                sem_search_searched_s[e][np.where(goal_neighbor_ss == 1)] = 1

            cn = infos[e]['goal_cat_id'] + 4
            wheres = np.where(wheres_delete_s[e])
            local_map[e, cn, :, :][wheres] = 0.0

        # Semantic Policy
        for e in range(num_scenes):
            ep_num = args.from_idx + traj_number[e] * num_scenes + e
            if next_step_dict_s[e]['next_goal'] and (not finished[e]):
                subgoal_counter_s[e] += 1

                full_map[e, :, lmb[e, 0]:lmb[e, 1], lmb[e, 2]:lmb[e, 3]] = \
                    local_map[e]
                full_pose[e] = local_pose[e] + \
                    torch.from_numpy(origins[e]).to(device).float()

                locs = full_pose[e].cpu().numpy()
                r, c = locs[1], locs[0]
                loc_r, loc_c = [int(r * 100.0 / args.map_resolution),
                                int(c * 100.0 / args.map_resolution)]

                lmb[e] = get_local_map_boundaries((loc_r, loc_c),
                                                  (local_w, local_h),
                                                  (full_w, full_h))

                planner_pose_inputs[e, 3:] = lmb[e]
                origins[e] = [lmb[e][2] * args.map_resolution / 100.0,
                              lmb[e][0] * args.map_resolution / 100.0, 0.]

                local_map[e] = full_map[e, :,
                                        lmb[e, 0]:lmb[e, 1], lmb[e, 2]:lmb[e, 3]]
                local_pose[e] = full_pose[e] - \
                    torch.from_numpy(origins[e]).to(device).float()

                goal_name = list_of_actions_s[e][list_of_actions_pointer_s[e]][0]

                obstacles = np.rint(local_map[e][0].cpu().numpy())
                if obstacles[120, 120] == 0:
                    mask = np.zeros((240, 240))
                    connected_regions = skimage.morphology.label(1 - obstacles, connectivity=2)
                    connected_lab = connected_regions[120, 120]
                    mask[np.where(connected_regions == connected_lab)] = 1
                    mask[np.where(skimage.morphology.binary_dilation(
                        obstacles, skimage.morphology.square(5)))] = 1
                else:
                    dilated = skimage.morphology.binary_dilation(
                        obstacles, skimage.morphology.square(5))
                    mask = skimage.morphology.convex_hull_image(
                        dilated).astype(float)
                mask_grid = into_grid(torch.tensor(mask), args.grid_sz)
                where_ones = len(torch.where(mask_grid)[0]) # shows the
                mask_grid = mask_grid.numpy().flatten()

                if (args.sem_policy_type == "none") or (args.explore_prob == 1.0):
                    chosen_i = local_rngs[e].choice(len(np.where(mask)[0]))
                    x_240 = np.where(mask)[0][chosen_i]
                    y_240 = np.where(mask)[1][chosen_i]
                    global_goals[e] = [x_240, y_240]

                else:
                    # Just reconst the common map save objects
                    map_reconst = torch.zeros(
                        (4+len(large_objects2idx), 240, 240))
                    map_reconst[:4] = local_map[e][:4]
                    test_see = {}
                    map_reconst[4+large_objects2idx['SinkBasin']
                                ] = local_map[e][4+1]
                    test_see[1] = 'SinkBasin'

                    start_idx = 2
                    for cat, catid in large_objects2idx.items():
                        if not (cat == 'SinkBasin'):
                            map_reconst[4+large_objects2idx[cat]
                                        ] = local_map[e][4+start_idx]
                            test_see[start_idx] = cat
                            start_idx += 1

                    if args.save_pictures and (not skip_save_pic[e]):
                        pics_dname = os.path.join(
                            "pictures", args.eval_split, args.dn, str(ep_num))
                    else:
                        pics_dname = None

                    steps_taken = next_step_dict_s[e]['steps_taken']
                    if (goal_name in all_objects2idx) or ("sem_search_all" in args.mlm_options):
                        if ("mixed_search" in args.mlm_options) and (subgoal_counter_s[e] > 5):
                            pred_probs = Unet_model_equal(map_reconst.unsqueeze(0).to(
                                device), target_name=goal_name, out_dname=pics_dname,
                                steps_taken=steps_taken, temperature=args.mlm_temperature)
                        else:
                            sem_temperature = subgoal_counter_s[e] if ("temperature_annealing" in args.mlm_options) else args.mlm_temperature
                            pred_probs = Unet_model(map_reconst.unsqueeze(0).to(
                                device), target_name=goal_name, out_dname=pics_dname,
                                steps_taken=steps_taken, temperature=sem_temperature)

                        # TODO: integrate the contents of this if-statements to sem_map_model.py
                        if isinstance(Unet_model, MLM):
                            # do not search where we have already searched before
                            pred_probs = pred_probs.detach().cpu()
                            pred_probs *= into_grid(torch.tensor(1 - sem_search_searched_s[e]), args.grid_sz)
                            pred_probs = pred_probs.numpy().flatten()
                        else:
                            pred_probs = pred_probs.view(73, -1)
                            pred_probs = softmax(pred_probs)
                            pred_probs = pred_probs.detach().cpu().numpy()
                            pred_probs = pred_probs[all_objects2idx[goal_name]]

                        if not pred_probs.size:  # If pred_probs is empty
                            pred_probs = mask_grid * 1 / float(where_ones)

                        pred_probs = (1-args.explore_prob) * pred_probs + \
                            args.explore_prob * mask_grid * \
                            1 / float(where_ones)

                    else:
                        pred_probs = mask_grid * 1 / float(where_ones)

                    # Now sample one index
                    pred_probs = pred_probs.astype('float64')
                    pred_probs = pred_probs.reshape(args.grid_sz ** 2)

                    # TODO: incorporate subgoal counter with argmax_prob?
                    argmax_prob = 1.0 if ("search_argmax_100" in args.mlm_options) else 0.5
                    if ("search_argmax" in args.mlm_options) and (local_rngs[e].rand() < argmax_prob):
                        pred_probs_2d = pred_probs.reshape((args.grid_sz, args.grid_sz))
                        max_x, max_y = searchArgmax(1, pred_probs_2d)

                        # center the obtained coordinates so that the sum of pred_probs is maximized
                        # for the square region of args.sem_search_del_size
                        del_sz = args.sem_search_del_size // 2
                        search_mask = np.zeros_like(pred_probs_2d)
                        search_mask[max(0, max_x - del_sz):min(240, max_x + del_sz + 1),
                                    max(0, max_y - del_sz):min(240, max_y + del_sz + 1)] = 1
                        chosen_cell_x, chosen_cell_y = searchArgmax(
                            args.sem_search_del_size, pred_probs_2d, mask=search_mask)

                        # chosen_cell_x, chosen_cell_y = searchArgmax(
                            # args.sem_search_del_size // 2, pred_probs.reshape((args.grid_sz, args.grid_sz)))
                    else:
                        pred_probs = pred_probs / np.sum(pred_probs)
                        chosen_cell = local_rngs[e].multinomial(1, pred_probs.tolist()) # error occurred here
                        chosen_cell = np.where(chosen_cell)[0][0]
                        chosen_cell_x = int(chosen_cell / args.grid_sz)
                        chosen_cell_y = chosen_cell % args.grid_sz

                    # Sample among this mask
                    mask_new = np.zeros((240, 240))
                    shrink_sz = 240 // args.grid_sz
                    mask_new[chosen_cell_x*shrink_sz:chosen_cell_x*shrink_sz+shrink_sz,
                                chosen_cell_y*shrink_sz:chosen_cell_y*shrink_sz+shrink_sz] = 1
                    mask_new = mask_new * mask * (1 - sem_search_searched_s[e])

                    if np.sum(mask_new) == 0:
                        chosen_i = local_rngs[e].choice(len(np.where(mask)[0]))
                        x_240 = np.where(mask)[0][chosen_i]
                        y_240 = np.where(mask)[1][chosen_i]

                    else:
                        chosen_i = local_rngs[e].choice(
                            len(np.where(mask_new)[0]))
                        x_240 = np.where(mask_new)[0][chosen_i]
                        y_240 = np.where(mask_new)[1][chosen_i]

                    if args.save_pictures and (not skip_save_pic[e]):
                        os.makedirs(pics_dname, exist_ok=True)
                        with open(os.path.join(pics_dname, f"{ep_num}.txt"), "a") as f:
                            f.write(
                                f"{steps_taken},{goal_name},{chosen_cell_x},{chosen_cell_y},{x_240},{y_240},{subgoal_counter_s[e]}\n")
                        Unet_model.plotSample(
                            pred_probs.reshape(
                                (1, args.grid_sz, args.grid_sz)),
                            os.path.join(pics_dname, "goal_sem_pol", f"{steps_taken}.html"), names=[goal_name], wrap_sz=1,
                            zmax=0.01)

                    global_goals[e] = [x_240, y_240]

        # ------------------------------------------------------------------
        # Take action and get next observation
        found_goal = [0 for _ in range(num_scenes)]
        goal_maps = [np.zeros((local_w, local_h)) for _ in range(num_scenes)]

        for e in range(num_scenes):
            goal_maps[e][global_goals[e][0], global_goals[e][1]] = 1

        for e in range(num_scenes):
            cn = infos[e]['goal_cat_id'] + 4
            prev_cns[e] = cn
            cat_semantic_map = local_map[e, cn, :, :].cpu().numpy()
            ep_num = args.from_idx + traj_number[e] * num_scenes + e
            if (not finished[e]) and args.save_pictures and (not skip_save_pic[e]):
                pics_dname = os.path.join("pictures", args.eval_split, args.dn, str(ep_num))
                target_pic_dname = os.path.join(pics_dname, "Sem_Map_Target")
                os.makedirs(target_pic_dname, exist_ok=True)
                steps_taken = next_step_dict_s[e]['steps_taken']
                cv2.imwrite(os.path.join(target_pic_dname, f"Sem_Map_Target_{steps_taken}.png"), cat_semantic_map * 255)

            if cat_semantic_map.sum() != 0.:
                new_goal_needed = args.delete_from_map_after_move_until_visible and (next_step_dict_s[e]['next_goal'] or next_step_dict_s[e]['delete_lamp'])
                # print('next_step_dict_s[e]:', next_step_dict_s[e])
                if new_goal_needed or (found_subgoal_coordinates[e] is None):
                    cat_semantic_scores = np.zeros_like(cat_semantic_map)
                    cat_semantic_scores[cat_semantic_map > 0] = 1.

                    # delete coordinates in which the search failed
                    delete_coords = np.where(wheres_delete_s[e])
                    cat_semantic_scores[delete_coords] = 0

                    # TODO: might be better if cat_semantic_scores is eroded first, so that the goal won't be at edge of the receptacle region
                    wheres_y, wheres_x = np.where(cat_semantic_scores)
                    if len(wheres_x) > 0:
                        # go to the location where the taget is observed the most
                        target_y, target_x = searchArgmax(
                            args.goal_search_del_size, cat_semantic_scores)
                        found_subgoal_coordinates[e] = (target_y, target_x)

                if found_subgoal_coordinates[e] is None:
                    if args.delete_from_map_after_move_until_visible or args.delete_pick2:
                        found_goal[e] = 0
                        goal_spotted_s[e] = False
                else:
                    goal_maps[e] = np.zeros_like(cat_semantic_map)
                    goal_maps[e][found_subgoal_coordinates[e]] = 1
                    found_goal[e] = 1
                    goal_spotted_s[e] = True

            else:
                found_subgoal_coordinates[e] = None
                if args.delete_from_map_after_move_until_visible or args.delete_pick2:
                    found_goal[e] = 0
                    goal_spotted_s[e] = False

        manual_step = None
        if args.manual_control:
            manual_step = input("Manual control ON. ENTER next agent step (a: RotateLeft, w: MoveAhead, d: RotateRight, u: LookUp, n: LookDown)")

        planner_inputs = [{} for e in range(num_scenes)]
        for e, p_input in enumerate(planner_inputs):
            p_input['map_pred'] = local_map[e, 0, :, :].cpu().numpy() #
            p_input['exp_pred'] = local_map[e, 1, :, :].cpu().numpy()
            p_input['pose_pred'] = planner_pose_inputs[e]
            p_input['goal'] = goal_maps[e]
            p_input['found_goal'] = found_goal[e]
            p_input['wait'] = finished[e]
            p_input['list_of_actions'] = list_of_actions_s[e]
            p_input['list_of_list_of_actions'] = list_of_list_of_actions_s[e]
            p_input['list_of_actions_pointer'] = list_of_actions_pointer_s[e]
            p_input['consecutive_interaction'] = consecutive_interaction_s[e]
            p_input['consecutive_target'] = target_instance_s[e]
            p_input['manual_step'] = manual_step
            if args.visualize or args.print_images:
                local_map[e, -1, :, :] = 1e-5
                p_input['sem_map_pred'] = local_map[e, 4:, :,
                                                    :].argmax(0).cpu().numpy()

            if first_steps[e]:
                p_input['consecutive_interaction'] = None
                p_input['consecutive_target'] = None

        obs, rew, done, infos, goal_success_s, next_step_dict_s = envs.plan_act_and_preprocess(
            planner_inputs, goal_spotted_s)
        goal_success_s = list(goal_success_s)
        view_angles = []

        current_obj_in_view = next_step_dict_s[0]['visible_objects']
        holding_something = next_step_dict_s[0]['holding_object']
        # print('holding', holding_something)
        # print('i see:', current_obj_in_view)
        for object in current_obj_in_view:
            if object not in seen_objects:
                seen_objects[object] = 1
            else:
                seen_objects[object] += 1

        for e, p_input in enumerate(planner_inputs):
            next_step_dict = next_step_dict_s[e]

            view_angle = next_step_dict['view_angle']
            view_angles.append(view_angle)

            num_steps_so_far[e] = next_step_dict['steps_taken']
            first_steps[e] = False

            fails[e] += next_step_dict['fails_cur']
            if args.leaderboard and fails[e] >= args.max_fails:
                print("Interact API failed %d times" % fails[e])
                task_finish[e] = True

            if not(args.no_pickup) and (args.map_mask_prop != 1 or args.no_pickup_update) and next_step_dict['picked_up'] and goal_success_s[e]:
                do_not_update_cat_s[e] = infos[e]['goal_cat_id']
            elif not(next_step_dict['picked_up']):
                do_not_update_cat_s[e] = None

        sem_map_module.set_view_angles(view_angles)
        for e, p_input in enumerate(planner_inputs):

            print('num_steps_so_far:', num_steps_so_far[e])
            print('num_errors_so_far:', fails[e])
            ################### INITIALIZE RANDOM EXPLORATION ################

            prev_goal_name = list_of_actions_s[e][list_of_actions_pointer_s[e]][0]
            if num_steps_so_far[e] < explore_steps and prev_goal_name == 'None':
                print('Exploring my environment...')

            if num_steps_so_far[e] >= explore_steps and prev_goal_name == 'None':
                print('Done exploring, reordering action sequence...')
                # first_actionSeq = list_of_list_of_actions[0]
                task_type = actions_dicts[e]['task_type']
                goal_instr = actions_dicts[e]['goal_instr']
                print(list_of_list_of_actions_s[e])
                reordered_sequences = get_reordered_actionSeq(list_of_list_of_actions_s[e], seen_objects, task_type, goal_instr)
                done_seq = [len(reordered_sequences[0]) - 1]
                # if len(reordered_sequences) > 1:
                #     done_seq.append(done_seq[0] + len(reordered_sequences[1]))
                # if len(reordered_sequences) > 2:
                #     done_seq.append(done_seq[1] + len(reordered_sequences[2]))

                if len(reordered_sequences) > 1:
                    filtered_actionSeq = reordered_sequences[0] # take the first actions
                    num_action_seq = 1

                else:
                    filtered_actionSeq = [action for sublist in reordered_sequences for action in sublist]
                # num_action_seq = 3
                print('seen counts:', seen_objects)
                print('filtered_actionSeq:', filtered_actionSeq)
                # if there are more than 3 seq actions, take the first 3, and prune the rest
                    
                list_of_actions_s[e] = filtered_actionSeq
                subgoal_steps[e] = 0

                # subgoal_counter_s[e] = 0
                # found_subgoal_coordinates[e] = None
                p_input['list_of_actions'] = list_of_actions_s[e]
                list_of_actions_pointer_s[e] = 0
                p_input['list_of_actions_pointer'] = list_of_actions_pointer_s[e]
                goal_name = list_of_actions_s[e][list_of_actions_pointer_s[e]][0]
                reset_goal_true_false = [False] * num_scenes
                reset_goal_true_false[e] = True

                returned, target_instance_s[e] = determine_consecutive_interx(
                    list_of_actions_s[e], list_of_actions_pointer_s[e]-1, whether_sliced_s[e])
                returned = False
                if returned:
                    consecutive_interaction_s[e] = list_of_actions_s[e][list_of_actions_pointer_s[e]][1]
                infos = envs.reset_goal(
                    reset_goal_true_false, goal_name, [None])
                goal_spotted_s[e] = False
                found_goal[e] = 0
                wheres_delete_s[e] = np.zeros((240, 240))
                sem_search_searched_s[e] = np.zeros((240, 240))
                next_action = None
                seen_receptacles = {obj: score for obj, score in seen_objects.items() if obj in receptacles}
                most_prominent_receptacle = max(seen_receptacles, key=seen_receptacles.get)

            ##################################
            # print('most_prominent_receptacle:', most_prominent_receptacle)
            subgoal_steps[e] += 1
            print('subgoal_steps:', subgoal_steps[e])
            if p_input['wait'] == 1 or next_step_dict_s[e]['keep_consecutive']:
                pass
            else:
                consecutive_interaction_s[e], target_instance_s[e] = None, None

            if goal_success_s[e]:
                subgoal_steps[e] = 0
                if not(fails[e] <= 0):
                    fails[e] -= 1
                # print("goal success")
                if list_of_actions_pointer_s[e] == len(list_of_actions_s[e]) - 1:
                    all_completed[e] = True
                else:
                    print('moving on to next goal...')
                    # print('list_of_actions_pointer_s[e]:', list_of_actions_pointer_s[e], 'done_seq:', done_seq) 
                    if done_seq != None and list_of_actions_pointer_s[e] in done_seq:
                        done_seq = done_seq[1:]
                        if holding_something: #if agent is holding something, place it at the most prominent receptacle
                            print(f'hands full, placing object on the {most_prominent_receptacle}...')
                            goal_name = most_prominent_receptacle
                            putdown = (goal_name, 'PutObject')
                            list_of_actions_s[e].insert(list_of_actions_pointer_s[e]+1, putdown)
                            done_seq = [x + 2 for x in done_seq]

                    # segment that moves to the next action if there are still actions left
                    list_of_actions_pointer_s[e] += 1
                    goal_name = list_of_actions_s[e][list_of_actions_pointer_s[e]][0]

                    reset_goal_true_false = [False] * num_scenes
                    reset_goal_true_false[e] = True

                    returned, target_instance_s[e] = determine_consecutive_interx(
                        list_of_actions_s[e], list_of_actions_pointer_s[e]-1, whether_sliced_s[e])
                    if returned:
                        consecutive_interaction_s[e] = list_of_actions_s[e][list_of_actions_pointer_s[e]][1]
                    infos = envs.reset_goal(
                        reset_goal_true_false, goal_name, consecutive_interaction_s)
                    goal_spotted_s[e] = False
                    found_goal[e] = 0
                    wheres_delete_s[e] = np.zeros((240, 240))
                    sem_search_searched_s[e] = np.zeros((240, 240))

            # print('available_subgoal_steps:', int(args.max_episode_length / num_action_seq))
            if subgoal_steps[e] >= int(args.max_episode_length / num_action_seq):  # Threshold, e.g., 200 steps 150 works with ep 102
                print(f"Taking too long to find the target. Moving to the next action...")
                prev_goal_name = list_of_actions_s[e][list_of_actions_pointer_s[e]][0]

                # Reset the sub-counter
                subgoal_steps[e] = 0
                # Move to the next action if there are more actions
                if list_of_actions_pointer_s[e] < len(list_of_actions_s[e]) - 1:
                    if holding_something and done_seq != None and list_of_actions_pointer_s[e] in done_seq: #if agent is holding something, place it at the most prominent receptacle
                        print(f'hands full, placing object on the {most_prominent_receptacle}...')
                        goal_name = most_prominent_receptacle
                        putdown = (goal_name, 'PutObject')
                        list_of_actions_s[e].insert(list_of_actions_pointer_s[e]+1, putdown)
                        done_seq = done_seq[1:]
                        done_seq = [x + 2 for x in done_seq]

                    ##added this
                    list_of_actions_pointer_s[e] += 1
                    goal_name = list_of_actions_s[e][list_of_actions_pointer_s[e]][0]

                    reset_goal_true_false = [False] * num_scenes
                    reset_goal_true_false[e] = True

                    returned, target_instance_s[e] = determine_consecutive_interx(
                        list_of_actions_s[e], list_of_actions_pointer_s[e]-1, whether_sliced_s[e])
                    returned = False
                    if returned:
                        consecutive_interaction_s[e] = list_of_actions_s[e][list_of_actions_pointer_s[e]][1]
                    infos = envs.reset_goal(
                        reset_goal_true_false, goal_name, [None])
                    goal_spotted_s[e] = False
                    found_goal[e] = 0
                    wheres_delete_s[e] = np.zeros((240, 240))
                    sem_search_searched_s[e] = np.zeros((240, 240))
                    next_action = None
                    ##added this
                    while goal_name == prev_goal_name and list_of_actions_pointer_s[e] != len(list_of_actions_s[e]) - 1: # for consecutive interactions, of the same object, skip them if goal is not found within
                        print(f"Moving to the next action because the previous goal was not found.") # the threshold, e.g., 200 steps. not possible for any actions to be closeobject if target is not found in the first place
                        list_of_actions_pointer_s[e] += 1
                        prev_goal_name = goal_name
                        goal_name = list_of_actions_s[e][list_of_actions_pointer_s[e]][0]

                        reset_goal_true_false = [False] * num_scenes
                        reset_goal_true_false[e] = True

                        returned, target_instance_s[e] = determine_consecutive_interx(
                            list_of_actions_s[e], list_of_actions_pointer_s[e]-1, whether_sliced_s[e])
                        returned = False
                        if returned:
                            consecutive_interaction_s[e] = list_of_actions_s[e][list_of_actions_pointer_s[e]][1]
                        infos = envs.reset_goal(
                            reset_goal_true_false, goal_name, [None])
                        goal_spotted_s[e] = False
                        found_goal[e] = 0
                        wheres_delete_s[e] = np.zeros((240, 240))
                        sem_search_searched_s[e] = np.zeros((240, 240))
                        # next_action = list_of_actions_s[e][list_of_actions_pointer_s[e]+1][1]
                        print('action:', list_of_actions_s[e][list_of_actions_pointer_s[e]][1])
                else:
                    # Handle the case when there are no more actions left
                    print(f"No more actions left for episode.")
                    all_completed[e] = True
        # ------------------------------------------------------------------
        # End episode and log
        for e in range(num_scenes): #condtion for ending the episode: skipped indices
            number_of_this_episode = args.from_idx + \
                traj_number[e] * num_scenes + e
            if number_of_this_episode in skip_indices:
                seen_objects = {}
                task_finish[e] = True

        failed_msg_printed = False
        for e in range(num_scenes):
            # evaluate the episode if success or not
            # if done_seq != None:
            #     if list_of_actions_pointer_s[e] in done_seq:    
            #         log_entry, success = envs.evaluate(e)
            #         log_entry, success = log_entry[0], success[0]
            #         # if not(finished[e] or all_completed[e]) and success:
            #         if success:
            #             print("This episode is a Success!")
            #             task_finish[e] = True
            #         elif not(finished[e]) and not success and goal_success_s[e] and not failed_msg_printed:
            #             print('this task is either not yet completed or failed, will continue looking!')
            #             failed_msg_printed = True
            #     #     print("This episode is a Failure! continuing the episode")
            #     #     task_finish[e] = False
            #     #     list_of_actions.add
            #             # continue the episode
            if all_completed[e]:
                if not(finished[e]) and args.test:
                    print("This episode is probably Success!")
                seen_objects = {}
                task_finish[e] = True

        # for e in range(num_scenes): #condition for ending the episode: all completed
        #     if list_of_actions_pointer_s[e] == 10: #start evaluating the episode if the agent has taken at least one action

        #         log_entry, success = envs.evaluate(e)
        #         log_entry, success = log_entry[0], success[0]
        #         if all_completed[e]:
        #             if not(finished[e]) and args.test: #if the episode is not finished and it is a test
        #                 print("This episode is probably Success!")
        #             if success:
        #                 task_finish[e] = True
        #             else:
        #                 print("No more actions. This episode is a Failure!")
        #                 task_finish[e] = True

        for e in range(num_scenes):
            if num_steps_so_far[e] >= args.max_episode_length and not(finished[e]):
                print("This outputted")
                task_finish[e] = True

        for e in range(num_scenes):
            number_of_this_episode = args.from_idx + \
                traj_number[e] * num_scenes + e
            if task_finish[e] and not(finished[e]) and not(number_of_this_episode in skip_indices):
                logname = "results/logs/log_" + args.eval_split + "_from_" + str(args.from_idx) + "_to_" + str(args.to_idx) + "_" + dn + ".txt"
                with open(logname, "a") as f:
                    number_of_this_episode = args.from_idx + \
                        traj_number[e] * num_scenes + e
                    f.write("\n")
                    f.write("===================================================\n")
                    f.write("episode # is " +
                            str(number_of_this_episode) + "\n")
                    # Write the goal instructions
                    
                    # f.write("")

                    for log in next_step_dict_s[e]['logs']:
                        f.write(log + "\n")

                    if all_completed[e]:
                        if not(finished[e]) and args.test:
                            f.write("This episode is probably Success!\n")

                    if not(args.test):
                        # success is  (True,), log_entry is ({..}, )
                        log_entry, success = envs.evaluate(e) # function call to evaluate the agent
                        log_entry, success = log_entry[0], success[0]
                        print("success is ", success)
                        f.write("success is " + str(success) + "\n")
                        print("log entry is " + str(log_entry))
                        f.write("log entry is " + str(log_entry) + "\n")
                        if success:
                            successes.append(log_entry)
                        else:
                            failures.append(log_entry)

                        print("saving success and failures for episode # ",
                              number_of_this_episode, "and process number is", e)
                        with open("results/successes/" + args.eval_split + "_successes_from_" + str(args.from_idx) + "_to_" + str(args.to_idx) + "_" + dn + ".p", "wb") as g:
                            pickle.dump(successes, g)
                        with open("results/fails/" + args.eval_split + "_failures_from_" + str(args.from_idx) + "_to_" + str(args.to_idx) + "_" + dn + ".p", "wb") as h:
                            pickle.dump(failures, h)

                    else:
                        print("episode # ", number_of_this_episode,
                              "ended and process number is", e)

                    if args.leaderboard and args.test:
                        actseq = next_step_dict_s[e]['actseq']
                        actseqs.append(actseq)

                # Add to analyze recs
                analyze_dict = {'task_type': actions_dicts[e]['task_type'], 'errs': next_step_dict_s[e]['errs'], 'action_pointer': list_of_actions_pointer_s[e], 'goal_found': goal_spotted_s[e],
                                'number_of_this_episode': number_of_this_episode}
                if not(args.test):
                    analyze_dict['success'] = envs.evaluate(e)[1][0]
                else:
                    analyze_dict['success'] = all_completed[e]
                analyze_recs.append(analyze_dict)
                with open("results/analyze_recs/" + args.eval_split + "_anaylsis_recs_from_" + str(args.from_idx) + "_to_" + str(args.to_idx) + "_" + dn + ".p", "wb") as iii:
                    pickle.dump(analyze_recs, iii)

                granularity_map = {"gt": "gt", "high": "noappended", "high_low": "appended"}
                granularity = granularity_map[args.language_granularity]

                logs_for_shell = f"results/shell/log_{args.eval_split}_{granularity}.txt"
                with open(logs_for_shell, "a") as f:
                    f.write("===================================================\n")
                    f.write("episode # is " + str(number_of_this_episode) + "\n")
                    f.write("goal instruction is to" + str(actions_dicts[e]['goal_instr']) + "\n")
                    f.write("number of steps taken is " + str(num_steps_so_far[e]) + "\n")
                    f.write("list of actions are " + str(list_of_actions_s[e]) + "\n")
                    if all_completed[e]:
                        if not(finished[e]) and args.test:
                            f.write("success is probably True" + "\n")

                    # f.write("number of completed goal conditions is " + str(log_entry['completed_goal_conditions']) + "\n")
                    # f.write('goal condition success rate is ' + str(log_entry['goal_condition_success']) + "\n")
                    # f.write("path length weighted success is " + str(log_entry['path_len_weighted_success_spl']) + "\n")
                    # f.write("path length weighted goal condition success is " + str(log_entry['path_len_weighted_goal_condition_spl']) + "\n")
                    # f.write("log entry is " + str(log_entry) + "\n")
                    f.write("===================================================\n")



if __name__ == "__main__":
    main()
    print("All finsihed!")

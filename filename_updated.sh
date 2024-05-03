#!/bin/bash


declare -A rand_nums  # For new episodes
declare -A completed_episodes  # For already completed episodes

# Read the completed episodes from the file and store them in the completed_episodes associative array
while read -ra eps; do
    for ep in "${eps[@]}"; do
        completed_episodes[$ep]=1
        rand_nums[$ep]=1  # Consider completed episodes as part of the total
    done
done < "completed_episodes.txt"

# Generate unique random episodes, including the completed ones, until total reaches 600
while [ ${#rand_nums[@]} -lt 600 ]; do
    ep=$((RANDOM % 820))
    rand_nums[$ep]=1  # This also avoids duplicates among newly generated numbers
done

echo "Generating ${#rand_nums[@]} episodes, including completed ones."
iteration=${#completed_episodes[@]}  # Start off from the number of completed episodes
output_file="your_output_file_path_here"  # Define your output file path
> "$output_file"  # Clear the contents of the file or create it

# Iterate only through the newly generated episodes, excluding the ones read from the file
for ep in "${!rand_nums[@]}"; do
    if [[ -z ${completed_episodes[$ep]} ]]; then  # Check if episode was not in the completed list
        iteration=$((iteration + 1))  # Increment iteration starting from the count of completed episodes
        echo "Iteration $iteration, Episode: $ep"
        echo "$ep" >> "$output_file"
	python main_shell.py -n1 --max_episode_length 1000 --num_local_steps 25 --num_processes 1 --eval_split valid_seen --from_idx $ep --to_idx $((ep + 1)) --max_fails 10 --debug_local --learned_depth --use_sem_seg --set_dn testrun -v 0 --which_gpu 0 --x_display 1 --sem_policy_type mlm --mlm_fname mlmscore_gpt --mlm_options aggregate_sum sem_search_all spatial_norm temperature_annealing new_obstacle_fn --seed 1 --splits alfred_data_small/splits/oct21.json --grid_sz 240 --mlm_temperature 1 --approx_last_action_success --language_granularity high --centering_strategy local_adjustment --target_offset_interaction 0.5 --obstacle_selem 9 --debug_env

         # Append the newly processed episode to the completed_episodes.txt file
        echo "$ep" >> "completed_episodes.txt"
    fi
done

echo "Completed generating episodes. Total iterations: $iteration."

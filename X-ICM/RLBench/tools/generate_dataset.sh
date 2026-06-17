#!/bin/bash

rlbench_tasks=(
    "open_drawer"
    "slide_block_to_color_target"
    "sweep_to_dustpan_of_size"
    "meat_off_grill"
    "turn_tap"
    "put_item_in_drawer"
    "close_jar"
    "reach_and_drag"
    "stack_blocks"
    "light_bulb_in"
    "put_money_in_safe"
    "place_wine_at_rack_location"
    "put_groceries_in_cupboard"
    "place_shape_in_shape_sorter"
    "push_buttons"
    "insert_onto_square_peg"
    "stack_cups"
    "place_cups"
)


zeroshot_tasks=(
    "basketball_in_hoop"
    "close_box"
    "close_laptop_lid"
    "empty_dishwasher"
    "get_ice_from_fridge"
    "hockey"
    "meat_on_grill"
    "move_hanger"
    "scoop_with_spatula"
    "setup_chess"
    "slide_block_to_target"
    "straighten_rope"
    "turn_oven_on"
    "wipe_desk"
)



tasks=("${zeroshot_tasks[@]}")

start=0  
length=1
tasks=("${tasks[@]:start:length}")

echo "${tasks[@]}"

# amount=200 #10
# splits=("train")

# for split in "${splits[@]}"; do
#     save_path="data/jiaming_private/roboprompt_zeroshot/$split"
#     for task in "${tasks[@]}"; do
#         xvfb-run -a python RLBench/tools/dataset_generator.py --tasks=$task \
#                             --save_path=$save_path \
#                             --renderer=opengl \
#                             --episodes_per_task=$amount \
#                             --processes=1 \
#                             --all_variations=False \
#                             --variations=1 \
        
#         mv $save_path/$task/variation0 $save_path/$task/all_variations
#     done
# done

amount=25
# amount=10
splits=("test")

for split in "${splits[@]}"; do
    # save_path="data/jiaming_private/roboprompt_rlbench/$split"
    # save_path="data/jiaming_private/roboprompt_zeroshot/$split"
    save_path="data/jiaming_private/roboprompt_zeroshot/$split"
    for task in "${tasks[@]}"; do
        xvfb-run -a python RLBench/tools/dataset_generator.py --tasks=$task \
                            --save_path=$save_path \
                            --renderer=opengl \
                            --episodes_per_task=$amount \
                            --processes=1 \
                            --all_variations=False \
                            --variations=1 \
                            
        mv $save_path/$task/variation0 $save_path/$task/all_variations
    done
done










# amount=10
# splits=("train")

# tasks_per_thread=3
 
# run_task() {
#     task="$1"
#     echo "Processing task: $task"
#     save_path="data/jiaming_private/roboprompt_zeroshot/$split"
#     xvfb-run -a python RLBench/tools/dataset_generator.py --tasks=$task \
#                             --save_path=$save_path \
#                             --renderer=opengl \
#                             --episodes_per_task=$amount \
#                             --processes=1 \
#                             --all_variations=False \
#                             --variations=1 \

#     mv $save_path/$task/variation0 $save_path/$task/all_variations
# }

# echo "Starting multi-threaded execution..."
# echo "Each thread will process $tasks_per_thread tasks."
# echo

# task_count=${#tasks[@]}
# threads=6
# current_task=0

# for ((i=0; i<threads; i++)); do
#   {
#     for ((j=0; j<tasks_per_thread; j++)); do
#       if ((current_task < task_count)); then
#         run_task "${tasks[current_task]}" &
#         ((current_task++))
#       fi
#     done
#     wait
#   } &
# done
 
# wait
# echo "All tasks completed."

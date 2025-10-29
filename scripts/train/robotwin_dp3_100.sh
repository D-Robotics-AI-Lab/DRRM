#!/bin/bash
config_dir=configs/dp3/dp3.yaml
task_list=(
    'block_hammer_beat' 'bottle_adjust' 'container_place'
    'dual_bottles_pick_hard' 'put_apple_cabinet'
    'tool_adjust' 'pick_apple_messy' 'dual_bottles_pick_easy' 
    'diverse_bottles_pick' 'empty_cup_place' 'shoe_place' 
    'dual_shoes_place' 'blocks_stack_easy' 'block_handover'
)
demo=100

for task in "${task_list[@]}"; do
    accelerate launch\
        --config_file configs/accelerate_config.yaml \
        main.py \
        --config-path="${config_dir%/*}" \
        --config-name="${config_dir##*/}" \
        train_dataset.path=datasets/drrm_robotwin1.0_D435_200_pcd \
        train_dataset.task=$task \
        train_dataset.demo=$demo
done
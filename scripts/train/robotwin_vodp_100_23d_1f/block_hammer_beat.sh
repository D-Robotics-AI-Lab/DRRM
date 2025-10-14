#!/bin/bash
config_dir=configs/vodp_train/vodp_23d_1f.yaml
task='block_hammer_beat'
demo=100

for task in "${task_list[@]}"; do
    accelerate launch\
        --config_file configs/accelerate_config.yaml \
        main.py \
        --config-path="${config_dir%/*}" \
        --config-name="${config_dir##*/}" \
        train_dataset.path=datasets/lerobot_D435_200 \
        train_dataset.task=$task \
        train_dataset.demo=$demo
done
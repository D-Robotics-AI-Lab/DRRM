#!/bin/bash
dataset=PATH/TO/YOUR/DATASET
task=YOUR/TASK/NAME
demo=null # The number of demonstrations used, defaults to all.
config_dir=configs/vodp/vodp_23d_1frame.yaml

accelerate launch\
    --config_file configs/accelerate_config.yaml \
    main.py \
    --config-path="${config_dir%/*}" \
    --config-name="${config_dir##*/}" \
    train_dataset.path=$dataset \
    train_dataset.task=$task \
    train_dataset.demo=$demo
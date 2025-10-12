#!/bin/bash
dataset=PATH/TO/YOUR/DATASET
task=YOUR/TASK/NAME
demo=null # The number of demonstrations used, defaults to all.

accelerate launch\
    --config_file configs/accelerate_config.yaml \
    main.py \
    --config-path=configs/vodp \
    --config-name=vodp_23d_1frame.yaml \
    train_dataset.path=$dataset \
    train_dataset.task=$task \
    train_dataset.demo=$demo
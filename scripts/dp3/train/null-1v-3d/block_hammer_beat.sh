#!/bin/bash
config=dp34thanos_1v_3d
task=place_cube_limit_3d
demo=null
path="${config/_*/}_${task}_${demo}_${config#*_}"

accelerate launch\
    --config_file configs/accelerate_config.yaml \
    main.py \
    --config-path=configs/dp3_train \
    --config-name=$config.yaml \
    train_dataset.task=$task \
    train_dataset.demo=$demo

~/bcecmd bos cp -r checkpoints/$path bos:/dg-algo/zehao.ni/checkpoints/$path
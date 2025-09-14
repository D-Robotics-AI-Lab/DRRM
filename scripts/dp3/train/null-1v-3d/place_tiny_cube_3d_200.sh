#!/bin/bash
config=dp34thanos_1v_3d
task=place_tiny_cube_3d_200
demo=null
path="${config/_*/}_${task}_None_${config#*_}"

accelerate launch\
    --config_file configs/accelerate_config.yaml \
    main.py \
    --config-path=configs/dp3_train \
    --config-name=$config.yaml \
    train_dataset.task=$task \
    train_dataset.demo=$demo \
    model.use_pc_color=true

~/bcecmd bos cp -r checkpoints/$path bos:/dg-algo/zehao.ni/checkpoints/$path
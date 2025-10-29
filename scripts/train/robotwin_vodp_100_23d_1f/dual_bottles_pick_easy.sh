#!/bin/bash
config_dir=configs/vodp_train/vodp_23d_1f.yaml
task='dual_bottles_pick_easy'
demo=100

accelerate launch\
    --config_file configs/accelerate_config.yaml \
    main.py \
    --config-path="${config_dir%/*}" \
    --config-name="${config_dir##*/}" \
    train_dataset.path=datasets/drrm_robotwin1.0_D435_200_rgb \
    train_dataset.task=$task \
    train_dataset.demo=$demo

config_name="${config_dir##*/}"
config="${config_name%.*}"
ckpt_name="${config/_*/}_${task}_${demo}_${config#*_}"
cp -r ./checkpoints/$ckpt_name /data/tos/users/zehao.ni/checkpoints/$ckpt_name
# rm -rf ./checkpoints/$ckpt_name
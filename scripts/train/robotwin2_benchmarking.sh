#!/bin/bash
config_dir=$1
task=$2
demo=100

accelerate launch\
    --config_file configs/accelerate_config.yaml \
    main.py \
    --config-path="${config_dir%/*}" \
    --config-name="${config_dir##*/}" \
    train_dataset.path=datasets/drrm_robotwin2.0_D435_Agilex_200_rgb \
    train_dataset.task=$task \
    train_dataset.demo=$demo

config_name="${config_dir##*/}"
config="${config_name%.*}"
ckpt_name="${config/_*/}_${task}_${demo}_${config#*_}"
echo ./checkpoints/$ckpt_name

if [ -n "$BACKUP_HOME" ] && [ -d "$BACKUP_HOME/checkpoints" ]; then
    cp -r "./checkpoints/$ckpt_name" "$BACKUP_HOME/checkpoints/$ckpt_name"
    echo "Backup completed: $ckpt_name has been copied to $BACKUP_HOME/checkpoints/"
else
    echo "Warning: BACKUP_HOME is not set or checkpoints directory does not exist"
    exit 1
fi
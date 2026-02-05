#!/bin/bash
config_dir=$1
task=$2
num_processes=$3
demo=100

if [ -n "$task" ]; then
    task_list=("$task")
else
    task_list=(
        'adjust_bottle' 'beat_block_hammer'
        'click_alarmclock' 'click_bell'
        'blocks_ranking_rgb'
    )
fi

num_flag=""
if [ -n "$num_processes" ]; then
    num_flag="--num_processes=$num_processes"
fi

for task in "${task_list[@]}"; do
    echo "Training task: $task with config: $config_dir"
    accelerate launch\
        --config_file configs/accelerate_config.yaml \
        $num_flag \
        main.py \
        --config-path="${config_dir%/*}" \
        --config-name="${config_dir##*/}" \
        train_dataset.path=datasets/drrm_robotwin2.0_D435_Agilex_200_rgb \
        train_dataset.task=$task \
        train_dataset.demo=$demo
    
    config_name="${config_dir##*/}"
    config="${config_name%.*}"
    ckpt_name="${config/_*/}/${task}_${demo}/${config#*_}"
    # ckpt_name="${config/_*/}"
    echo ./checkpoints/$ckpt_name

    if [ -n "$BACKUP_HOME" ] && [ -d "$BACKUP_HOME/checkpoints" ]; then
        cp -r "./checkpoints/$ckpt_name" "$BACKUP_HOME/checkpoints/$ckpt_name"
        echo "Backup completed: $ckpt_name has been copied to $BACKUP_HOME/checkpoints/"
    else
        echo "Warning: BACKUP_HOME is not set or checkpoints directory does not exist"
        exit 1
    fi
done


#!/bin/bash
model_config=$1

task_list=(
    'adjust_bottle' 'beat_block_hammer'
    'click_alarmclock' 'click_bell'
    'blocks_ranking_rgb'
    # 'blocks_ranking_size'
)
config_list=(
    'agilex_config' 'agilex_config_mid' 'agilex_config_hard'
)
demo=100

policy="${config/_*/}"
for task in "${task_list[@]}"; do
    for config in "${config_list[@]}"; do
        echo $task $config $model_config
        ./scripts/eval/eval_robotwin2.0_parallel.sh VAR0 $task $config $model_config $demo 16
    done
done
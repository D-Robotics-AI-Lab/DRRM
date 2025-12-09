#!/bin/bash
model_config=$1

task_list=(
    'adjust_bottle' 'beat_block_hammer' 'blocks_ranking_rgb'
    'blocks_ranking_size' 'click_alarmclock' 'click_bell'
)
demo=100

policy="${config/_*/}"
for task in "${task_list[@]}"; do
    echo $task VODPP
    ./scripts/eval/eval_robotwin2.0_parallel.sh VODPP $task agilex_config $model_config $demo 4
done
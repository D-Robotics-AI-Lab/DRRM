#!/bin/bash
task_list=(
    # 'adjust_bottle' 'beat_block_hammer' 'blocks_ranking_rgb'
    'blocks_ranking_size' 'click_alarmclock' 'click_bell'
)
demo=100

policy="${config/_*/}"
for task in "${task_list[@]}"; do
    echo $task VODPP
    ./scripts/eval/eval_robotwin2.0_parallel.sh VODPP $task agilex_config largedit_fm_23d_1f $demo 4
done
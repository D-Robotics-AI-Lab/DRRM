#!/bin/bash
task_list=(
    'adjust_bottle' 'blocks_ranking_rgb' 'blocks_ranking_size'
    'click_alarmclock' 'click_bell'
)
# 'beat_block_hammer'

cd simulation/robotwin2.0/
for task in "${task_list[@]}"; do
    echo "Collecting data for task: $task"
    bash collect_data.sh $task agilex_config 0
done
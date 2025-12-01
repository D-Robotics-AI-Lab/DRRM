#!/bin/bash
config=vodp_23d_1f
task_list=(
    'beat_block_hammer'
)
demo=100

policy="${config/_*/}"
for task in "${task_list[@]}"; do
    echo $task $policy $ckpt_name
    ./scripts/eval/eval_robotwin2.0.sh VODP $task agilex_config 23d_1f
done
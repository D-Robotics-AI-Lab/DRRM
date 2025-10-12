#!/bin/bash
config=vodp_23d_1f
task_list=(
    'block_hammer_beat' 'bottle_adjust' 'container_place'
    'dual_bottles_pick_hard' 'put_apple_cabinet'
    'tool_adjust' 'pick_apple_messy' 'dual_bottles_pick_easy' 
    'diverse_bottles_pick' 'empty_cup_place' 'shoe_place' 
    'dual_shoes_place' 'blocks_stack_easy' 'block_handover'
)
demo=100

for task in "${task_list[@]}"; do
    ckpt_name="${config/_*/}_${task}_${demo}_${config#*_}"
    echo $task $ckpt_name
    ./scripts/mvdp/eval/evaluation.sh $task $ckpt_name
done
#!/bin/bash
config=dp
task_list=(
    'block_hammer_beat' 'bottle_adjust' 'container_place'
    'dual_bottles_pick_hard' 'put_apple_cabinet'
    'tool_adjust' 'pick_apple_messy' 'dual_bottles_pick_easy' 
    'diverse_bottles_pick' 'empty_cup_place' 'shoe_place' 
    'dual_shoes_place' 'blocks_stack_easy' 'block_handover'
)
demo=100

policy="${config/_*/}"
for task in "${task_list[@]}"; do
    ckpt_name="${config/_*/}_${task}_${demo}_${config#*_}"
    echo $task $policy $ckpt_name
    ./scripts/eval/eval_robotwin.sh $task $policy $ckpt_name
done
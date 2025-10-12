#!/bin/bash
config=dp_baseline

# task_list=(
#     'block_hammer_beat' 'bottle_adjust' 'container_place' 
#     'dual_bottles_pick_hard' 'put_apple_cabinet'
# )

task_list=(
    # 'block_handover' 'blocks_stack_easy' 'diverse_bottles_pick' 
    # 'dual_bottles_pick_easy' 'dual_shoes_place' 'empty_cup_place'
    'pick_apple_messy' 'shoe_place' 'tool_adjust'
)

for task in "${task_list[@]}"; do
    demo=100
    path="${config/_*/}_${task}_${demo}_${config#*_}"
    echo $task $demo $path
    ./scripts/mvdp/eval/evaluation.sh $config $task $demo $path
done
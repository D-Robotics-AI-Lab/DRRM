#!/bin/bash
config=mvdp_1v_23d_23_pool_single_frame

# task_list=(
#     'block_hammer_beat' 'bottle_adjust' 'container_place' 
#     'dual_bottles_pick_hard' 'put_apple_cabinet'
# )

# task_list=(
#     'tool_adjust' 'pick_apple_messy' 'dual_bottles_pick_easy' 
#     'diverse_bottles_pick' 'empty_cup_place' 'shoe_place' 
#     'dual_shoes_place' 'blocks_stack_easy' 'block_handover'
# )

task_list=(
    'tool_adjust'
    'pick_apple_messy'
    'block_hammer_beat'
    'dual_bottles_pick_easy'
    'bottle_adjust'
    'diverse_bottles_pick'
    'container_place'
    'dual_bottles_pick_hard'
    'empty_cup_place'
    'shoe_place'
    'put_apple_cabinet'
    'blocks_stack_easy'
    'block_handover'
    'dual_shoes_place'
)

for task in "${task_list[@]}"; do
    demo=100
    path="${config/_*/}_${task}_${demo}_${config#*_}"
    echo $task $demo $path
    ./scripts/mvdp/eval/evaluation.sh $config $task $demo $path
done
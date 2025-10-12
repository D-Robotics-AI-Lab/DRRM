#!/bin/bash
config=mvdp_1v_3d_23_pool

# task_list=(
#     'block_hammer_beat' 'bottle_adjust' 'container_place' 
#     'dual_bottles_pick_hard' 'put_apple_cabinet'
# )

task_list=(
    'pick_apple_messy' 'dual_bottles_pick_easy' 'blocks_stack_easy'
)

for task in "${task_list[@]}"; do
    demo=100
    path="${config/_*/}_${task}_${demo}_${config#*_}"
    echo $task $demo $path
    ./scripts/mvdp/eval/evaluation.sh $config $task $demo $path
done
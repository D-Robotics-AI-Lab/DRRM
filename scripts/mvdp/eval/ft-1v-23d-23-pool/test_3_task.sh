#!/bin/bash
config=mvdp_1v_23d_23_pool

task_list=(
    'bottle_adjust' 'container_place' 'diverse_bottles_pick'
)

for task in "${task_list[@]}"; do
    demo=100
    path="${config/_*/}_${task}_ft_${config#*_}"
    echo $task $demo $path
    ./scripts/mvdp/eval/evaluation.sh $config $task $demo $path
done
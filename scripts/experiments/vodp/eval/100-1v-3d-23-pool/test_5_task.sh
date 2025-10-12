#!/bin/bash
config=mvdp_1v_3d_23_pool

task=block_hammer_beat
demo=100
path="${config/_*/}_${task}_${demo}_${config#*_}"
echo $task $demo $path
./scripts/mvdp/eval/evaluation.sh $config $task $demo $path

# task=bottle_adjust
# demo=100
# path="${config/_*/}_${task}_${demo}_${config#*_}"
# echo $task $demo $path
# ./scripts/mvdp/eval/evaluation.sh $config $task $demo $path

# task=container_place
# demo=100
# path="${config/_*/}_${task}_${demo}_${config#*_}"
# echo $task $demo $path
# ./scripts/mvdp/eval/evaluation.sh $config $task $demo $path

# task=dual_bottles_pick_hard
# demo=100
# path="${config/_*/}_${task}_${demo}_${config#*_}"
# echo $task $demo $path
# ./scripts/mvdp/eval/evaluation.sh $config $task $demo $path

# task=put_apple_cabinet
# demo=100
# path="${config/_*/}_${task}_${demo}_${config#*_}"
# echo $task $demo $path
# ./scripts/mvdp/eval/evaluation.sh $config $task $demo $path
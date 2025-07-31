#!/bin/bash
config=mvdp_1v_3d_23_global
task=block_hammer_beat
demo=100
path="${config/_*/}_${task}_${demo}_${config#*_}"

./scripts/mvdp/eval/evaluation.sh $config $task $demo $path
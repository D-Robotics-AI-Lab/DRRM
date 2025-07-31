#!/bin/bash
config=mvdp_2v_3d_23_low_dim
task=dual_bottles_pick_hard
demo=100
path="${config/_*/}_${task}_${demo}_${config#*_}"

./scripts/mvdp/eval/evaluation.sh $config $task $demo $path
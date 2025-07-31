#!/bin/bash
config=mvdp_1v_23d_23_low_dim
task=container_place
demo=100
path="${config/_*/}_${task}_${demo}_${config#*_}"

./scripts/mvdp/eval/evaluation.sh $config $task $demo $path
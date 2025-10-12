#!/bin/bash
config=mvdp_2v_3d
task=bottle_adjust
demo=100
path="${config/_*/}_${task}_${demo}_${config#*_}"

./scripts/mvdp/eval/evaluation.sh $config $task $demo $path
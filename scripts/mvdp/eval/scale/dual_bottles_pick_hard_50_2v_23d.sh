#!/bin/bash
config=mvdp_2v_23d
task=dual_bottles_pick_hard
demo=50
path="${config/_*/}_${task}_${demo}_${config#*_}"

./scripts/mvdp/eval/evaluation.sh $config $task $demo $path
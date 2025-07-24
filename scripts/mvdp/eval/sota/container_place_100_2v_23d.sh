#!/bin/bash
config=mvdp_2v_23d
task=container_place
demo=100
path="${config/_*/}_${task}_${demo}_${config#*_}"

./scripts/mvdp/eval/evaluation.sh $config $task $demo $path
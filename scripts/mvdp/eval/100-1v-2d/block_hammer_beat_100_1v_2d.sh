#!/bin/bash
config=mvdp_1v_2d
task=block_hammer_beat
demo=100
path="${config/_*/}_${task}_${demo}_${config#*_}"

./scripts/mvdp/eval/evaluation.sh $config $task $demo $path
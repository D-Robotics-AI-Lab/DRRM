#!/bin/bash
config=dp_baseline
task=put_apple_cabinet
demo=100
path="${config/_*/}_${task}_${demo}_${config#*_}"

./scripts/mvdp/eval/evaluation.sh $config $task $demo $path
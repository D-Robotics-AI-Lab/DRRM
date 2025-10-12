#!/bin/bash
task=$1
policy=$2
ckpt_name=$3
ckpt_dir="checkpoints/$ckpt_name"

for ((i=1; i<=3; i+=1)); do
    exp_num=exp$i
    save_dir="eval_result/$ckpt_name/$exp_num"

    python eval_sim/robotwin/script/eval_policy_$policy.py \
        --checkpoint-dir $ckpt_dir \
        --save-dir $save_dir \
        --task-name $task \
        --num-process 8 \
        --seed 0
    
    while IFS= read -r -d '' file; do
        # 提取纯文件名（不含路径）
        filename=$(basename "$file")
        souc=$save_dir/$filename
        dest="${save_dir%/$exp_num}/${exp_num}_${filename}"
        echo "保留 $exp_num 结果：$souc -> $dest"
        cp $souc $dest
    done < <(find "$save_dir" -maxdepth 1 -type f -print0)
done
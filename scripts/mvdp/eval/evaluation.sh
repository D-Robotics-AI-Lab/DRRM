#!/bin/bash
config=$1
task=$2
demo=$3
path=$4

echo config: $config
echo task: $task
echo demo: $demo
echo path: $path

ckp_path="checkpoints/$path"
if [ -d "$ckp_path" ]; then
    abs_path="$(realpath "$ckp_path" 2>/dev/null || echo "转换失败")"
    echo "模型绝对路径: $abs_path"
else
    echo "模型拷贝：/bosspace/$ckp_path/ -> $ckp_path"
    # ~/bcecmd bos cp -r bos:/dg-algo/zehao.ni/$ckp_path/ $ckp_path
    ~/bcecmd bos sync bos:/dg-algo/zehao.ni/$ckp_path/ $ckp_path/ \
        --exclude "bos:/dg-algo/zehao.ni/$ckp_path/ema/*" \
        --exclude "bos:/dg-algo/zehao.ni/$ckp_path/checkpoint*/*"
    abs_path="$(realpath "$ckp_path" 2>/dev/null || echo "转换失败")"
    echo "模型绝对路径: $abs_path"
fi

for ((i=1; i<=3; i+=1)); do
    exp_num=exp$i
    save_path="eval_result/$path/$exp_num"

    python eval_sim/robotwin/script/eval_policy_dp.py \
        --config-name configs/mvdp_train/$config.yaml \
        --checkpoint-dir $ckp_path \
        --save-dir $save_path \
        --task-name $task \
        --num-process 8 \
        --seed 0
    ~/bcecmd bos cp -r $save_path/ bos:/dg-algo/zehao.ni/$save_path
    
    while IFS= read -r -d '' file; do
        # 提取纯文件名（不含路径）
        filename=$(basename "$file")
        souc=$save_path/$filename
        dest="${save_path%/$exp_num}/${exp_num}_${filename}"
        echo "保留 $exp_num 结果：$souc -> $dest"
        cp $souc $dest
    done < <(find "$save_path" -maxdepth 1 -type f -print0)
    rm -rf $save_path
done

echo "清除模型: checkpoints/$path"
rm -rf checkpoints/$path
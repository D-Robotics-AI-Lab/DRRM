
task_list=('block_hammer_beat' 'put_apple_cabinet' 'dual_bottles_pick_hard' 'bottle_adjust'
            'container_place' 'block_handover' 'blocks_stack_easy' 'blocks_stack_hard' 
            'diverse_bottles_pick' 'dual_bottles_pick_easy' 'dual_shoes_place' 'empty_cup_place'
            'mug_hanging_easy' 'mug_hanging_hard' 'pick_apple_messy' 'shoe_place' 'tool_adjust')

# task_list=('block_hammer_beat')

for task in "${task_list[@]}"; do
    echo $task

    task_pkl=${task}_D435_pkl_200
    pkl_path="datasets/robotwin/$task_pkl"
    echo $pkl_path
    if [ -d "$pkl_path" ]; then
        abs_path="$(realpath "$pkl_path" 2>/dev/null || echo "转换失败")"
        echo "绝对路径: $abs_path"
    else
        echo "数据拷贝：/bosspace/$pkl_path/ -> $pkl_path"
        ~/bcecmd bos cp -r bos:/dg-algo/fa.fu/$pkl_path/ $pkl_path
        abs_path="$(realpath "$pkl_path" 2>/dev/null || echo "转换失败")"
        echo "绝对路径: $abs_path"
    fi

    task_lerobot="${task}_D435_200"
    lerobot_path="datasets/lerobot3d/$task_lerobot"
    echo $lerobot_path
    python scripts/robotwin/robotwin2lerobot4dp3.py \
        --task $task\
        --src_dir ./$pkl_path \
        --dst_dir ./$lerobot_path \
        --repo D-robotics/$task_lerobot \
        --fps 30
    ~/bcecmd bos cp -r $lerobot_path/ bos:/dg-algo/zehao.ni/$lerobot_path

    rm -rf $pkl_path
done
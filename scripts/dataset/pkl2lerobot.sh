echo "当前路径: $(pwd)"

task_list=('beat_block_hammer')

process_task() {
    task=$1
    echo "$task"

    task_pkl=${task}/data
    pkl_path="datasets/robotwin2.0_D435_Agilex_200_rgb/$task_pkl"
    echo $pkl_path
    if [ -d "$pkl_path" ]; then
        abs_path="$(realpath "$pkl_path" 2>/dev/null || echo "转换失败")"
        echo "绝对路径: $abs_path"
    else
        echo "路径不存在"
        return 1
    fi

    task_lerobot="${task}"
    lerobot_path="datasets/drrm_robotwin2.0_D435_Agilex_200_rgb/$task_lerobot"
    echo $lerobot_path
    python tools/preprocess_robotwin2.py \
        --task $task\
        --src_dir $pkl_path \
        --dst_dir ./$lerobot_path \
        --repo D-robotics/$task_lerobot \
        --fps 30
}

export -f process_task

# 使用GNU Parallel并行处理，最多同时运行4个进程
echo "${task_list[@]}" | tr ' ' '\n' | parallel -j 8 process_task {}
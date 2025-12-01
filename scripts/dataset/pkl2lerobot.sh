echo "当前路径: $(pwd)"

# task_list=('adjust_bottle' 'click_bell' 'hanging_mug' 'move_stapler_pad' 'place_a2b_left'       
#     'place_can_basket' 'place_fan' 'place_phone_stand' 'rotate_qrcode' 'stack_blocks_two'
#     'beat_block_hammer' 'dump_bin_bigbin' 'lift_pot' 'open_laptop' 'place_a2b_right'      
#     'place_cans_plasticbox' 'place_mouse_pad' 'place_shoe' 'scan_object' 'stack_bowls_three'
#     'blocks_ranking_rgb' 'grab_roller' 'move_can_pot' 'open_microwave' 'place_bread_basket'   
#     'place_container_plate' 'place_object_basket' 'press_stapler' 'shake_bottle' 'stack_bowls_two'
#     'blocks_ranking_size' 'handover_block' 'move_pillbottle_pad' 'pick_diverse_bottles' 'place_bread_skillet'  
#     'place_dual_shoes' 'place_object_scale' 'put_bottles_dustbin' 'shake_bottle_horizontally' 'stamp_seal'
#     'click_alarmclock' 'handover_mic' 'move_playingcard_away' 'pick_dual_bottles' 'place_burger_fries'   
#     'place_empty_cup' 'place_object_stand' 'put_object_cabinet' 'stack_blocks_three' 'turn_switch'
# )

task_list=(
    'adjust_bottle' 'blocks_ranking_rgb' 'blocks_ranking_size'
    'click_alarmclock' 'click_bell'
)
# 'beat_block_hammer'

process_task() {
    task=$1
    echo "$task"

    task_pkl=${task}/data
    pkl_path="../../datasets/robotwin2.0_D435_Agilex_200_rgb/$task_pkl"
    echo $pkl_path
    if [ -d "$pkl_path" ]; then
        abs_path="$(realpath "$pkl_path" 2>/dev/null || echo "转换失败")"
        echo "绝对路径: $abs_path"
    else
        echo "路径不存在"
        return 1
    fi

    task_lerobot="${task}"
    lerobot_path="../../datasets/drrm_robotwin2.0_D435_Agilex_200_rgb/$task_lerobot"
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
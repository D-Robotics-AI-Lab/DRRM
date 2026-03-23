#!/bin/bash
# task_list=(
#     'adjust_bottle' 'beat_block_hammer' 'blocks_ranking_rgb'
#       'blocks_ranking_size' 'click_alarmclock' 'click_bell'
# )
task_list=('hanging_mug' 'move_stapler_pad' 'place_a2b_left'       
    'place_can_basket' 'place_fan' 'place_phone_stand' 'rotate_qrcode' 'stack_blocks_two'
    'beat_block_hammer' 'dump_bin_bigbin' 'lift_pot' 'open_laptop' 'place_a2b_right'      
    'place_cans_plasticbox' 'place_mouse_pad' 'place_shoe' 'scan_object' 'stack_bowls_three'
    'grab_roller' 'move_can_pot' 'open_microwave' 'place_bread_basket'   
    'place_container_plate' 'place_object_basket' 'press_stapler' 'shake_bottle' 'stack_bowls_two'
    'handover_block' 'move_pillbottle_pad' 'pick_diverse_bottles' 'place_bread_skillet'  
    'place_dual_shoes' 'place_object_scale' 'put_bottles_dustbin' 'shake_bottle_horizontally' 'stamp_seal'
    'handover_mic' 'move_playingcard_away' 'pick_dual_bottles' 'place_burger_fries'   
    'place_empty_cup' 'place_object_stand' 'put_object_cabinet' 'stack_blocks_three' 'turn_switch'
)
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

cd simulation/robotwin2.0/
for task in "${task_list[@]}"; do
    echo "Collecting data for task: $task"
    bash collect_data.sh $task agilex_config 0
done
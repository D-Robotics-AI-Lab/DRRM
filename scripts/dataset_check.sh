task_list=(
    'block_hammer_beat' 'block_handover' 'blocks_stack_easy' 
    'bottle_adjust' 'container_place' 'diverse_bottles_pick' 
    'dual_bottles_pick_easy' 'dual_bottles_pick_hard' 'dual_shoes_place' 
    'empty_cup_place' 'pick_apple_messy' 'put_apple_cabinet' 'shoe_place' 'tool_adjust'
)

for task in "${task_list[@]}"; do
    echo ${task}
    ls -lah datasets/lerobot/${task}_D435_200
done
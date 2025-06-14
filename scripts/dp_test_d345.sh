python eval_sim/robotwin/script/eval_policy_dp.py \
    --config-name configs/dp_emvis_spatial_debug.yaml \
    --checkpoint-dir checkpoints/dp_dual_bottles_pick_hard_exp3_gate_v4_checkpoint-22000 \
    --save-dir /root/projects/RoboticsManipulation/eval_result/debug_gate_8gpu_128_d435/ \
    --task-name dual_bottles_pick_hard \
    --num-process 8 \
    --head-camera-type D435 \
    --seed 0
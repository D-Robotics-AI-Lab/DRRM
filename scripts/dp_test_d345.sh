python eval_sim/robotwin/script/eval_policy_dp.py \
    --config-name configs/dp_emvis_spatial_debug.yaml \
    --checkpoint-dir checkpoints/exp3_dim \
    --save-dir /root/projects/RoboticsManipulation/eval_result/debug_dim_8gpu_128_d435/ \
    --task-name dual_bottles_pick_hard \
    --num-process 8 \
    --head-camera-type D435 \
    --seed 0
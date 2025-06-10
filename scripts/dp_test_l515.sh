python eval_sim/robotwin/script/eval_policy_dp.py \
    --config-name configs/dp_baseline.yaml \
    --checkpoint-dir checkpoints/checkpoint-22000 \
    --save-dir /root/projects/RoboticsManipulation/eval_result/dp_baseline_0605_4gpu_128_l515/ \
    --task-name dual_bottles_pick_hard \
    --num-process 4 \
    --head-camera-type L515 \
    --seed 0
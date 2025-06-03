
# dp3 实验
- 训练
export HF_LEROBOT_HOME="/workspace/.cache/huggingface/lerobot"
export HF_HOME="/workspace/.cache/huggingface"
export HYDRA_FULL_ERROR=1 
CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch --num_processes=4 main.py --config-name=test_dp3_policy.yaml
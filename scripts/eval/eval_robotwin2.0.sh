#!/bin/bash

# == keep unchanged ==
policy_name=${1}
task_name=${2}
task_config=${3}
ckpt_setting=${4}
expert_data_num=${5}
seed=${6}
gpu_id=${7}
DEBUG=True

export CUDA_VISIBLE_DEVICES=${gpu_id}
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"

cd simulation/robotwin2.0

PYTHONWARNINGS=ignore::UserWarning \
python script/eval_policy.py --config policy/$policy_name/deploy_policy.yml \
    --overrides \
    --task_name ${task_name} \
    --task_config ${task_config} \
    --ckpt_setting ${ckpt_setting} \
    --expert_data_num ${expert_data_num} \
    --seed ${seed}
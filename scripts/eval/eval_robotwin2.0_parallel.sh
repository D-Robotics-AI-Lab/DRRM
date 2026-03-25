#!/bin/bash

# == keep unchanged ==
policy_name=${1}
task_name=${2}
task_config=${3}
ckpt_setting=${4}
expert_data_num=${5}
num_process=${6}
checkpoint_num=${7}
seed=${8}

# gpu_id=${8}
DEBUG=True

# export CUDA_VISIBLE_DEVICES=${gpu_id}
# echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"

cd simulation/robotwin2.0

# Build argument array so we can conditionally append --num_process only when provided.
args=(
    "script/eval_policy_parallel.py"
    "--config" "policy/$policy_name/deploy_policy.yml"
    "--overrides"
    "--task_name" "${task_name}"
    "--task_config" "${task_config}"
    "--ckpt_setting" "${ckpt_setting}"
    "--expert_data_num" "${expert_data_num}"
)

# Append optional overrides only if variables are non-empty
if [ -n "${num_process}" ]; then
    args+=("--num_process" "${num_process}")
fi

if [ -n "${checkpoint_num}" ]; then
    args+=("--checkpoint_num" "${checkpoint_num}")
fi

if [ -n "${seed}" ]; then
    args+=("--seed" "${seed}")
fi

PYTHONWARNINGS=ignore::UserWarning \
python "${args[@]}"
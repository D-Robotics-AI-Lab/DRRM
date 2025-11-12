# D-robotics Robotic Manipulation Platform
## 🔥 Latest Work: VO-DP
[![project page](https://img.shields.io/badge/Project%20Page-DRRM-orange)](https://d-robotics-ai-lab.github.io/vodp/)
[![arXiv paper](https://img.shields.io/badge/Paper-arXiv-red)](https://arxiv.org/abs/2510.15530)
[![dataset](https://img.shields.io/badge/Dataset-HuggingFace-yellow)](https://huggingface.co/datasets/D-Robotics/DRRM)

https://github.com/user-attachments/assets/fdca37aa-164b-4281-a446-3c909a3f1456

## 🌟 Update Log
- **2025-11**: The robotwin2.0 simulator is available now! Check out the [simulation/robotwin2.0](simulation/robotwin2.0/)
- **2025-10**: We release the code and dataset for our latest work [VO-DP: Semantic-Geometric Adaptive Diffusion Policy for Vision-Only Robotic Manipulation](https://arxiv.org/abs/2510.15530).

## ⚙️ Installation
### Basic Environment Setup
```bash
git clone https://github.com/D-Robotics-AI-Lab/DRRM.git
cd DRRM
conda create -n drrm python=3.10
conda activate drrm
pip install -e .
```

### VODP Environment Setup

```bash
mkdir -p third_party
cd third_party
git clone https://github.com/facebookresearch/vggt.git
cd vggt
pip install .
cd ../..
```

### Robotwin Environment Setup
See `simulation/robotwin/README.md` for detailed simulator installation and usage steps.

## 📊 Dataset Preparation
Create the `datasets/` directory and download one of the preprocessed dataset bundles from HuggingFace:

```bash
mkdir -p datasets
```

Preprocessed bundles available (choose one):
- [`drrm_robotwin1.0_D435_200_rgb`](https://huggingface.co/datasets/D-Robotics/DRRM/tree/main/drrm_robotwin1.0_D435_200_rgb) — RGB-only version (no point clouds)
- [`drrm_robotwin1.0_D435_200_pcd`](https://huggingface.co/datasets/D-Robotics/DRRM/tree/main/drrm_robotwin1.0_D435_200_pcd) — includes point clouds

Visit https://huggingface.co/datasets/D-Robotics/DRRM to download the dataset and place it under `datasets/`.

## 📑 Training
1. Modify the acceleration configuration file based on your training environment: [configs/accelerate_config.yaml](configs/accelerate_config.yaml)
2. In the training script [scripts/train_demo.sh](scripts/train_demo.sh), specify the following parameters:
   - `dataset`: Path to your training dataset
   - `task`: Your training task
   - `demo`: Number of demonstration samples to use (set to `null` for unlimited)
   - `config_dir`: Training configuration directory (refer to [configs/](configs/))

- Training VODP
```
accelerate launch\
    --config_file configs/accelerate_config.yaml \
    main.py \
    --config-path="configs/vodp_train" \
    --config-name="vodp_23d_1f.yaml" \
    train_dataset.path=datasets/lerobot_D435_200 \
    train_dataset.task=block_hammer_beat \
    train_dataset.demo=100
```

- Training DP
```
accelerate launch\
    --config_file configs/accelerate_config.yaml \
    main.py \
    --config-path="configs/dp_train" \
    --config-name="dp.yaml" \
    train_dataset.path=datasets/lerobot_D435_200 \
    train_dataset.task=block_hammer_beat \
    train_dataset.demo=100
```

- Training DP3
```
accelerate launch\
    --config_file configs/accelerate_config.yaml \
    main.py \
    --config-path="configs/dp3_train" \
    --config-name="dp3.yaml" \
    train_dataset.path=datasets/lerobot43d_D435_200 \
    train_dataset.task=block_hammer_beat \
    train_dataset.demo=100
```

<!-- ### Training Your Own Model
...... -->

## 🤖 Simulation Evaluation
DRRM is compatible with the Robotwin simulator — refer to the following files:

- `scripts/eval/eval_robotwin.sh` — convenience shell wrapper used for experiments.
- `scripts/eval/robotwin_exp/` — example experiment wrappers used in our paper.
- `simulation/robotwin/script/` — simulator-facing Python evaluation scripts (e.g. `eval_policy_vodp.py`, `eval_policy_dp.py`, `eval_policy_dp3.py`).

Supported benchmark tasks:

```
[
    'block_hammer_beat', 'bottle_adjust', 'container_place',
    'dual_bottles_pick_hard', 'put_apple_cabinet',
    'tool_adjust', 'pick_apple_messy', 'dual_bottles_pick_easy',
    'diverse_bottles_pick', 'empty_cup_place', 'shoe_place',
    'dual_shoes_place', 'blocks_stack_easy', 'block_handover'
]
```


### Basic usage
Replace `YOUR/CHECKPOINT/DIR` and `YOUR/SAVE/DIR` with the paths to your checkpoint directory and the directory where you want to store evaluation results. The `--num-process` flag controls parallel simulator workers.

```bash
# Evaluate VODP
python simulation/robotwin/script/eval_policy_vodp.py \
    --checkpoint-dir YOUR/CHECKPOINT/DIR \
    --save-dir YOUR/SAVE/DIR \
    --task-name TASK_NAME \
    --num-process 8 \
    --seed 0
```

```bash
# Evaluate DP
python simulation/robotwin/script/eval_policy_dp.py \
    --checkpoint-dir YOUR/CHECKPOINT/DIR \
    --save-dir YOUR/SAVE/DIR \
    --task-name TASK_NAME \
    --num-process 8 \
    --seed 0
```

```bash
# Evaluate DP3
python simulation/robotwin/script/eval_policy_dp3.py \
    --checkpoint-dir YOUR/CHECKPOINT/DIR \
    --save-dir YOUR/SAVE/DIR \
    --task-name TASK_NAME \
    --num-process 8 \
    --seed 0
```

## 👏 Citation
```
@article{ni2025vodp,
  title={VO-DP: Semantic-Geometric Adaptive Diffusion Policy for Vision-Only Robotic Manipulation},
  author={Zehao Ni and Yonghao He and Lingfeng Qian and Jilei Mao and Fa Fu and Wei Sui and Hu Su and Junran Peng and Zhipeng Wang and Bin He},
  journal={arXiv preprint arXiv:2510.15530},
  year={2025}
}
```

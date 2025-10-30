# D-robotics Robotic Manipulation Platform
## 🔥 Latest Work: VO-DP
[![project page](https://img.shields.io/badge/Project%20Page-GitHub-blue)](https://d-robotics-ai-lab.github.io/vodp/)
[![arXiv paper](https://img.shields.io/badge/arXiv-Paper-red)](https://arxiv.org/abs/2510.15530)
[![dataset](https://img.shields.io/badge/Dataset-DRRM-blue)](https://huggingface.co/datasets/D-Robotics/DRRM)

https://github.com/user-attachments/assets/fdca37aa-164b-4281-a446-3c909a3f1456


## ⚙️ Installation 

### Basic Environment Setup
```
git clone https://github.com/D-Robotics-AI-Lab/DRRM.git
# D-robotics Robotic Manipulation Platform
## 🔥 Latest Work: VO-DP
[![project page](https://img.shields.io/badge/Project%20Page-GitHub-blue)](https://d-robotics-ai-lab.github.io/vodp/)
[![arXiv paper](https://img.shields.io/badge/arXiv-Paper-red)](https://arxiv.org/abs/2510.15530)
[![dataset](https://img.shields.io/badge/Dataset-DRRM-blue)](https://huggingface.co/datasets/D-Robotics/DRRM)

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
```
mkdir third_party
cd third_party
git clone https://github.com/facebookresearch/vggt.git
cd vggt
pip install .
```

### Robotwin Environment Setup

See [Robotwin Usage Documentation](simulators/README.md) for setup instructions.

## 📊 Dataset Preparation
```
mkdir -p datasets
```
### Preprocessed Dataset Download
Refer to [D-Robotics/DRRM](https://huggingface.co/datasets/D-Robotics/DRRM) for preprocessed datasets:

- Download either `drrm_robotwin1.0_D435_200_pcd` or `drrm_robotwin1.0_D435_200_rgb` to your `datasets/` directory:
    - [drrm_robotwin1.0_D435_200_rgb (without point clouds)](https://huggingface.co/datasets/D-Robotics/DRRM/tree/main/drrm_robotwin1.0_D435_200_rgb)
    - [drrm_robotwin1.0_D435_200_pcd](https://huggingface.co/datasets/D-Robotics/DRRM/tree/main/drrm_robotwin1.0_D435_200_pcd)


### Preparing Your Own Dataset
......


## 📑 Training
1. Modify the acceleration configuration file based on your training environment: [configs/accelerate_config.yaml](configs/accelerate_config.yaml)
2. In the training script [scripts/train_demo.sh](scripts/train_demo.sh), specify the following parameters:
   - `dataset`: Path to your training dataset
   - `task`: Your training task
   - `demo`: Number of demonstration samples to use (set to `null` for unlimited)
   - `config_dir`: Training configuration directory (refer to [configs/](configs/))
   These parameters can also be set directly in the training command.

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

# The checkpoint is saved to `checkpoints/vodp_block_hammer_beat_100_23d_1f`
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

# The checkpoint is saved to `checkpoints/vodp_block_hammer_beat_100_23d_1f`
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

# The checkpoint is saved to `checkpoints/vodp_block_hammer_beat_100_23d_1f`
```

### Training Your Own Model
......

## 🤖 Simulation Evaluation
......

## 👏 Citation
```
@article{ni2025vodp,
  title={VO-DP: Semantic-Geometric Adaptive Diffusion Policy for Vision-Only Robotic Manipulation},
  author={Zehao Ni and Yonghao He and Lingfeng Qian and Jilei Mao and Fa Fu and Wei Sui and Hu Su and Junran Peng and Zhipeng Wang and Bin He},
  journal={arXiv preprint arXiv:2510.15530},
  year={2025}
}
```

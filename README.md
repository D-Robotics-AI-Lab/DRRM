# D-robotics Robotic Manipulation Platform
## Latest Work: VO-DP
| VO-DP: Semantic-Geometric Adaptive Diffusion Policy for Vision-Only Robotic Manipulation

<video src="https://drobotics-ailab.tos-cn-beijing.volces.com/users/zehao.ni/github/VO-DP-4k_30.mp4" controls="controls" width="500" height="300"></video>

## Installation

### Basic Environment Setup
```
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

### Dataset Preparation
```
mkdir -p datasets
```
#### Preprocessed Dataset Download
- [Robotwin1.0_200demos_RGB]() -> datasets/lerobot_D435_200
- [Robotwin1.0_200demos_Pcd]() -> datasets/lerobot43d_D435_200


#### Preparing Your Own Dataset
......


## Training
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

## Simulation Evaluation
......

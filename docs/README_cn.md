# RoboticsManipulation

## 环境安装
```
conda create -n robotics_manipulation python=3.10
conda activate robotics_manipulation

# install lerobot
git clone https://github.com/huggingface/lerobot.git
cd lerobot
pip install build
python -m build
pip install dist/lerobot-0.1.0-py3-none-any.whl

安装vggt
vggt中需要依赖numpy<2, 与lerobot冲突
git clone https://github.com/facebookresearch/vggt.git
cd vggt
pip install -e .
cd ..

安装RoboticsManipulation
git clone --recurse-submodules https://github.com/D-Robotics-AI-Lab/RoboticsManipulation.git
cd RoboticsManipulation
pip install -e .
cd ..





```
## 项目结构说明

## 数据处理
```
1. robotwin数据转lerobot数据
python scripts/robotwin/robotwin2lerobot4dp.py --src_dir data/dual_bottles_pick_easy_D435_pkl/ --dst_dir data/lerobot/dual_bottles_pick_easy_D435 --repo D-robotics/dual_bottles_pick_easy_D435

2. robotwin数据转lerobot数据, 增加vggt
python scripts/robotwin/robotwin2lerobot4dp_vggt.py --src_dir data/dual_bottles_pick_easy_D435_pkl/ --dst_dir data/lerobot/dual_bottles_pick_easy_D435_drrm --repo D-robotics/dual_bottles_pick_easy_D435_drrm --vggt_path checkpoints/VGGT-1B/model.pt
```
## 运行
```
accelerate launch main.py --config-name=test_policy.yaml
```
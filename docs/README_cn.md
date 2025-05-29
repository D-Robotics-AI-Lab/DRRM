# RoboticsManipulation

## 环境安装
```
conda create -n robotics_manipulation
conda activate robotics_manipulation

git clone https://github.com/D-Robotics-AI-Lab/RoboticsManipulation.git
cd RoboticsManipulation
pip install -e .

```
## 项目结构说明

## 数据处理
```
1. robotwin数据转lerobot数据
python script/robotwin/robotwin2lerobot4dp.py --src_dir data/dual_bottles_pick_easy_D435_pkl/ --dst_dir data/lerobot/dual_bottles_pick_easy_D435 --repo D-robotics/dual_bottles_pick_easy_D435

2. robotwin数据转lerobot数据, 增加vggt
python script/robotwin/robotwin2lerobot4dp_vggt.py --src_dir data/dual_bottles_pick_easy_D435_pkl/ --dst_dir data/lerobot/dual_bottles_pick_easy_D435_drrm --repo D-robotics/dual_bottles_pick_easy_D435_drrm --vggt_path checkpoints/VGGT-1B/model.pt
```
## 运行
```
accelerate launch main.py --config-name=test_policy.yaml
```
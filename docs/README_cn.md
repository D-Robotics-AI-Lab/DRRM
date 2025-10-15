# RoboticsManipulation

## 基础环境安装
```
conda create -n drrm python=3.10
conda activate drrm
pip install build

# install lerobot
cd $ROBOTICSMANIPULATION_HOME
git clone https://github.com/huggingface/lerobot.git
cd lerobot
git checkout 2b71789e15c35418b1ccecbceb81f4a598bfd883
<!-- pip install . -->
python -m build
pip install dist/lerobot-0.1.0-py3-none-any.whl

# install vggt
# vggt中需要依赖numpy<2, 与lerobot冲突
cd $ROBOTICSMANIPULATION_HOME
git clone https://github.com/facebookresearch/vggt.git
cd vggt
pip install .
下载 VGGT ckp 到 ./pretrained/VGGT-1B/model.pt <- bos:/dg-algo/zehao.ni/models/VGGT-1B/model.pt

$ install RoboticsManipulation
cd DRRM
git clone --recurse-submodules https://github.com/D-Robotics-AI-Lab/RoboticsManipulation.git
cd RoboticsManipulation
pip install -e .

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
accelerate launch main.py --config-name=dp_baseline.yaml
```
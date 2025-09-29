# RoboticsManipulation

## 环境安装
```
export ROBOTICSMANIPULATION_HOME=/path/of/roboticsmanipulation
conda create -n robotics_manipulation python=3.10
conda activate robotics_manipulation

# install lerobot
cd $ROBOTICSMANIPULATION_HOME
git clone https://github.com/huggingface/lerobot.git
cd lerobot
git checkout a445d9c9da6bea99a8972daa4fe1fdd053d711d2
pip install build
python -m build
pip install dist/lerobot-0.1.0-py3-none-any.whl

# install vggt
# vggt中需要依赖numpy<2, 与lerobot冲突
pip install numpy==1.26.4
pip install datasets=3.6.0

cd $ROBOTICSMANIPULATION_HOME
git clone https://github.com/facebookresearch/vggt.git
cd vggt
pip install -e .
下载 VGGT ckp 到 ./pretrained/VGGT-1B/model.pt <- bos:/dg-algo/zehao.ni/models/VGGT-1B/model.pt

$ install RoboticsManipulation
cd $ROBOTICSMANIPULATION_HOME
git clone --recurse-submodules https://github.com/D-Robotics-AI-Lab/RoboticsManipulation.git
cd RoboticsManipulation
pip install -e .

```
## robotwin安装
```
apt install libvulkan1 mesa-vulkan-drivers vulkan-tools

conda activate robotics_manipulation
pip install sapien==3.0.0b1 scipy==1.10.1 mplib==0.1.1 trimesh==4.4.3 open3d==0.18.0 openai

cd $ROBOTICSMANIPULATION_HOME
git clone https://github.com/facebookresearch/pytorch3d.git
cd pytorch3d
pip install -e .

下载aloha_urdf.zip && main_models.zip到asserts文件夹中


Modify mplib Library Code
3.1 Remove convex=True
# mplib.planner (mplib/planner.py) line 71
# remove `convex=True`

self.robot = ArticulatedModel(
            urdf,
            srdf,
            [0, 0, -9.81],
            user_link_names,
            user_joint_names,
            convex=True,
            verbose=False,
        )
=> 
self.robot = ArticulatedModel(
            urdf,
            srdf,
            [0, 0, -9.81],
            user_link_names,
            user_joint_names,
            # convex=True,
            verbose=False,
        )
3.2 Remove or collide
# mplib.planner (mplib/planner.py) line 848
# remove `or collide`

if np.linalg.norm(delta_twist) < 1e-4 or collide or not within_joint_limit:
                return {"status": "screw plan failed"}
=>
if np.linalg.norm(delta_twist) < 1e-4 or not within_joint_limit:
                return {"status": "screw plan failed"}
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
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
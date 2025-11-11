# RobotWin — Installation & Setup

This document describes the minimal steps to set up the RobotWin simulation environment used in this project.

## Overview

- Install system Vulkan drivers and tools.
- Create and activate the conda environment.
- Install Python dependencies (specific tested versions).
- Download required assets.
- Apply two small, local edits to the mplib planner to avoid runtime errors.

## Prerequisites

Install Vulkan and related drivers:
```bash
sudo apt update
sudo apt install -y libvulkan1 mesa-vulkan-drivers vulkan-tools
```

## Python environment

Activate your conda environment (replace `drrm` with your env name if different):
```bash
conda activate drrm
```

Install required Python packages (versions used in this project):
```bash
pip install sapien==3.0.0b1 scipy==1.10.1 mplib==0.1.1 trimesh==4.4.3 open3d==0.18.0 openai
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
```

## Download assets

From the robotwin root, create an assets directory and download:
```bash
cd simulation/robotwin
mkdir -p assets
cd assets
python ../script/download_asset.py
```

Unpack the downloaded zips:
```bash
unzip aloha_urdf.zip && rm -f aloha_urdf.zip
unzip main_models.zip && rm -f main_models.zip
```

## Required local edits to mplib

Two small edits to mplib avoid compatibility issues. Edit `mplib/planner.py` in your site-packages or your local copy.

1) Remove `convex=True` parameter when creating ArticulatedModel (around line ~71):
Before:
```py
self.robot = ArticulatedModel(
    urdf,
    srdf,
    [0, 0, -9.81],
    user_link_names,
    user_joint_names,
    convex=True,
    verbose=False,
)
```
After:
```py
self.robot = ArticulatedModel(
    urdf,
    srdf,
    [0, 0, -9.81],
    user_link_names,
    user_joint_names,
    # convex=True,
    verbose=False,
)
```

2) Remove `or collide` from the screw planning failure condition (around line ~848):
Before:
```py
if np.linalg.norm(delta_twist) < 1e-4 or collide or not within_joint_limit:
    return {"status": "screw plan failed"}
```
After:
```py
if np.linalg.norm(delta_twist) < 1e-4 or not within_joint_limit:
    return {"status": "screw plan failed"}
```

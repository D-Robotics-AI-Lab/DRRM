#!/usr/bin/env python
"""
Convert *.pkl episodes to LeRobot v2.1 dataset format with video support.

Usage:
python scripts/pkl2lerobot_dp3_camera.py \
    --root ./data/dual_bottles_pick_hard_D435_pkl \
    --repo D-robotics/dual_bottles_pick_hard_D435_camera \
    --episodes 50 \
    --fps 40
"""

import pickle
from pathlib import Path
from typing import Dict, Any

import cv2
import numpy as np
import tyro
from tqdm import tqdm
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# Constants
DEFAULT_IMAGE_SIZE = (320, 240)  # (width, height)
DEFAULT_FPS = 10
DEFAULT_EPISODES = 100
TASK_NAME = "dual bottles pick hard"

# Feature schema definition for the LeRobot dataset
FEATURES = {
    "head_cam": {
        "dtype": "video", 
        "shape": (240, 320, 3), 
        "names": ["h", "w", "c"]
    },
    "endpose": {
        "dtype": "float32", 
        "shape": (14,), 
        "names": ["endpose"]
    },
    "agent_pos": {
        "dtype": "float32", 
        "shape": (14,), 
        "names": ["agent_pos"]
    },
    "action": {
        "dtype": "float32", 
        "shape": (14,), 
        "names": ["action"]
    },
}


def load_pickle_data(pkl_path: Path) -> Dict[str, Any]:
    """Load and return data from a pickle file."""
    try:
        with pkl_path.open("rb") as f:
            return pickle.load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to load {pkl_path}: {e}")


def process_image(rgb_image: np.ndarray, target_size: tuple = DEFAULT_IMAGE_SIZE) -> np.ndarray:
    """Process and resize RGB image to target dimensions."""
    height, width = target_size[1], target_size[0]  # OpenCV uses (height, width)
    
    if rgb_image.shape[:2] != (height, width):
        rgb_image = cv2.resize(
            rgb_image, 
            target_size, 
            interpolation=cv2.INTER_LINEAR
        )
    
    return rgb_image


def extract_frame_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract and process frame data from pickle data."""
    try:
        # Extract image data
        rgb_image = data["observation"]["head_camera"]["rgb"]
        head_cam = process_image(rgb_image)
        
        # Extract robot state data
        joint_action = data["joint_action"].astype(np.float32)
        endpose = data["endpose"].astype(np.float32)
        
        return {
            "task": TASK_NAME,
            "head_cam": head_cam,
            "agent_pos": joint_action,
            "action": joint_action,
            "endpose": endpose,
        }
    except KeyError as e:
        raise ValueError(f"Missing required data field: {e}")
    except Exception as e:
        raise RuntimeError(f"Failed to extract frame data: {e}")


def process_episode(episode_dir: Path, dataset: LeRobotDataset) -> bool:
    """Process a single episode directory and add frames to dataset."""
    if not episode_dir.is_dir():
        return False
    
    # Get sorted pickle files
    pkl_files = sorted(episode_dir.glob("*.pkl"), key=lambda p: int(p.stem))
    
    if not pkl_files:
        print(f"[WARNING] No pickle files found in {episode_dir}")
        return False
    
    # Process each frame
    for pkl_path in tqdm(pkl_files, desc=f"Processing {episode_dir.name}", leave=False):
        try:
            data = load_pickle_data(pkl_path)
            frame = extract_frame_data(data)
            dataset.add_frame(frame)
        except Exception as e:
            print(f"[ERROR] Failed to process {pkl_path}: {e}")
            continue
    
    # Save episode
    dataset.save_episode()
    return True


def main(
    root: Path,
    repo: str,
    episodes: int = DEFAULT_EPISODES,
    fps: int = DEFAULT_FPS,
    push: bool = False,
) -> None:
    """
    Convert pickle episodes to LeRobot dataset format.
    
    Args:
        root: Root directory containing episode data
        repo: HuggingFace repository ID for dataset
        episodes: Number of episodes to process
        fps: Frames per second for dataset
        push: Whether to push dataset to HuggingFace
    """
    if not root.exists():
        raise FileNotFoundError(f"Root directory does not exist: {root}")
    
    print(f"Creating LeRobot dataset: {repo}")
    
    # Create LeRobot dataset
    dataset = LeRobotDataset.create(
        repo_id=repo,
        fps=fps,
        robot_type="AgileBot",
        features=FEATURES,
        image_writer_threads=4,
        use_videos=True,
    )
    
    # Process episodes
    processed_count = 0
    for ep_idx in range(episodes):
        episode_dir = root / f"episode{ep_idx}"
        
        if process_episode(episode_dir, dataset):
            processed_count += 1
            print(f"[INFO] Processed episode {ep_idx}")
        else:
            print(f"[INFO] Skipping episode {ep_idx} - directory not found or empty")
    
    print(f"[INFO] Successfully processed {processed_count} episodes")
    
    # Push to HuggingFace if requested
    if push:
        print("[INFO] Pushing dataset to HuggingFace...")
        dataset.push_to_hub(push_videos=True, tags=["emvis", "converted"])
        print("[INFO] Dataset pushed successfully")


if __name__ == "__main__":
    tyro.cli(main)

#!/usr/bin/env python3
"""
Convert *.pkl episodes to LeRobot v2.1 dataset with VGGT features saved separately.

This script processes pickle files containing robot demonstration data and converts them
to LeRobot dataset format. VGGT visual features are extracted and saved as .npy files.

Usage example:
    python scripts/pkl2lerobot_dp3_emvis.py \
          --root ./data/dual_bottles_pick_hard_D435_pkl \
          --repo D-robotics/dual_bottles_pick_hard_D435_emvis \
          --episodes 50 \
          --fps 40 \
          --vggt_path /workspace/RoboTwin/vggt_pretrain.pt 
        #   --push
"""

import os
import pickle
from pathlib import Path
from typing import Dict, Any, List

# Set environment variables before importing HF libraries
os.environ["HF_HOME"] = "/workspace/.cache/huggingface"
os.environ["HF_LEROBOT_HOME"] = "/workspace/.cache/huggingface/lerobot"

import cv2
import numpy as np
import torch
import tyro
from tqdm import tqdm

from emvis.vggt_encoder import VGGTEncoder
from emvis.utils import preprocess_images
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

# Constants
TASK_STR = "dual bottles pick hard"
TARGET_IMAGE_SIZE = (320, 240)  # (width, height)

# Feature schema definition
FEATURES = {
    # Camera data (saved as video files)
    "head_cam": {
        "dtype": "video",
        "shape": (240, 320, 3),
        "names": ["h", "w", "c"]
    },
    # Action and state data
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


def initialize_vggt_encoder(vggt_path: Path) -> VGGTEncoder:
    """Initialize and load the VGGT encoder model."""
    encoder = VGGTEncoder(intermediate_layer_idx=[4, 11, 17, 23])
    encoder.load_pretrained_model(str(vggt_path))
    encoder.to("cuda").eval()
    return encoder


def process_image(rgb: np.ndarray) -> np.ndarray:
    """Process and resize RGB image to target size."""
    if rgb.shape[:2] != TARGET_IMAGE_SIZE[::-1]:  # OpenCV uses (height, width)
        rgb = cv2.resize(rgb, TARGET_IMAGE_SIZE, interpolation=cv2.INTER_LINEAR)
    return rgb


def extract_vggt_features(rgb: np.ndarray, encoder: VGGTEncoder) -> Dict[str, torch.Tensor]:
    """Extract VGGT features from RGB image."""
    # Convert to tensor and preprocess
    img_t = torch.from_numpy(rgb.transpose(2, 0, 1) / 255.0).float().cuda()  # CHW
    img_t = preprocess_images(img_t.unsqueeze(0))  # (1, 3, H, W)
    
    # Extract features
    with torch.no_grad():
        vggt_features = encoder(img_t.unsqueeze(0))  # (1, 1, 3, H, W)
    
    torch.cuda.empty_cache()
    return vggt_features


def flatten_tensor(tensor: torch.Tensor) -> np.ndarray:
    """Convert tensor to flattened numpy array on CPU."""
    return tensor.squeeze().cpu().numpy()


def create_frame_dict(data: Dict[str, Any], vggt_features: Dict[str, torch.Tensor], 
                     rgb: np.ndarray) -> Dict[str, Any]:
    """Create frame dictionary with all required features."""
    # Extract basic data
    state = data["joint_action"].astype(np.float32)
    action = data["joint_action"].astype(np.float32)
    endpose = data["endpose"].astype(np.float32)
    
    # Create frame dictionary
    frame = {
        "task": TASK_STR,
        "head_cam": rgb,
        "agent_pos": state,
        "action": action,
        "endpose": endpose,
    }
    
    return frame


def update_feature_buffer(buffer: Dict[str, List], features: Dict[str, torch.Tensor]) -> None:
    """Update VGGT feature buffer with new features."""
    spatial_tokens = features['spatial_tokens_list'][4] + features['spatial_tokens_list'][11] + features['spatial_tokens_list'][17] + features['spatial_tokens_list'][23]
    feature_mappings = [

        ('camera_tokens_4', features['camera_tokens_list'][4]),
        ('camera_tokens_11', features['camera_tokens_list'][11]),
        ('camera_tokens_17', features['camera_tokens_list'][17]),
        ('camera_tokens_23', features['camera_tokens_list'][23]),
        ('image_tokens', features['image_tokens']),
        ('image_tokens_pos', features['image_tokens_pos']),
    ]
    
    for name, tensor in feature_mappings:
        buffer[name].append(flatten_tensor(tensor))


def save_features_to_disk(buffer: Dict[str, List], repo: str, ep_idx: int) -> None:
    """Save VGGT features buffer to disk as .npy files."""
    for key, value in buffer.items():
        output_dir = Path(f"/workspace/.cache/huggingface/lerobot/{repo}/npy/{key}")
        output_dir.mkdir(parents=True, exist_ok=True)
        try:
            np.save(output_dir / f"{ep_idx}.npy", value)
        except Exception as e:
            print(f"[ERROR] Could not save {key} for episode {ep_idx}: {e}")


def process_episode(ep_dir: Path, encoder: VGGTEncoder, ds: LeRobotDataset, 
                   ep_idx: int, repo: str) -> bool:
    """Process a single episode directory."""
    if not ep_dir.is_dir():
        print(f"[INFO] Directory {ep_dir} does not exist, skipping.")
        return False
    
    pkl_files = sorted(ep_dir.glob("*.pkl"), key=lambda p: int(p.stem))
    if not pkl_files:
        print(f"[WARNING] No .pkl files found in {ep_dir}")
        return False
    
    # Initialize VGGT features buffer
    vggt_features_buffer = {
        'spatial_tokens_4': [], 'spatial_tokens_11': [], 'spatial_tokens_17': [], 'spatial_tokens_23': [],
        'camera_tokens_4': [], 'camera_tokens_11': [], 'camera_tokens_17': [], 'camera_tokens_23': [],
        'image_tokens': [], 'image_tokens_pos': [],
    }
    
    try:
        for pkl_path in tqdm(pkl_files, desc=f"Episode {ep_idx}", leave=False):
            # Load pickle data
            with pkl_path.open("rb") as f:
                data = pickle.load(f)
            
            # Process image
            rgb = data["observation"]["head_camera"]["rgb"]  # uint8 (H, W, 3)
            rgb = process_image(rgb)
            
            # Extract VGGT features
            vggt_features = extract_vggt_features(rgb, encoder)
            
            # Create frame dictionary and add to dataset
            frame = create_frame_dict(data, vggt_features, rgb)
            ds.add_frame(frame)
            
            # Update feature buffer
            update_feature_buffer(vggt_features_buffer, vggt_features)
        
        # Save episode and features
        ds.save_episode()
        save_features_to_disk(vggt_features_buffer, repo, ep_idx)
        return True
        
    except Exception as e:
        print(f"[ERROR] Failed to process episode {ep_idx}: {e}")
        return False


def main(
    root: Path,
    repo: str,
    episodes: int = 50,
    fps: int = 40,
    vggt_path: Path = Path("/workspace/RoboTwin/vggt_pretrain.pt"),
    push: bool = False,
) -> None:
    """Main function to convert pickle episodes to LeRobot dataset."""
    print(f"[INFO] Starting conversion of {episodes} episodes from {root}")
    print(f"[INFO] Target repository: {repo}")
    print(f"[INFO] VGGT model path: {vggt_path}")
    
    # Initialize VGGT encoder
    encoder = initialize_vggt_encoder(vggt_path)
    
    # Create LeRobot dataset
    ds = LeRobotDataset.create(
        repo_id=repo,
        fps=fps,
        robot_type="AgileBot",
        features=FEATURES,
        use_videos=True,
        image_writer_threads=4,  # Async PNG writing for better I/O performance
    )
    
    # Process episodes
    episode_dirs = sorted(root.glob("episode*"))[:episodes]
    successful_episodes = 0
    
    for ep_idx, ep_dir in enumerate(episode_dirs):
        if process_episode(ep_dir, encoder, ds, ep_idx, repo):
            successful_episodes += 1
        
        if ep_idx + 1 >= episodes:
            break
    
    print(f"[INFO] Successfully processed {successful_episodes}/{episodes} episodes")
    
    # Push to hub if requested
    if push:
        print("[INFO] Pushing dataset to hub...")
        ds.push_to_hub(push_videos=True, tags=["emvis", "converted"])
        print("[INFO] Dataset pushed successfully!")
    
    print("[INFO] Conversion completed!")


if __name__ == "__main__":
    tyro.cli(main)

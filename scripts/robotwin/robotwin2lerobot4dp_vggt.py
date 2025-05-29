import os
import pickle
from pathlib import Path
from typing import Dict, Any, List
import cv2
import numpy as np
import torch
from tqdm import tqdm
import argparse
import shutil

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

from emvis.vggt_encoder import VGGTEncoder
from emvis.utils import preprocess_images


# Constants
TASK_STR = "dual bottles pick easy"
CAMERA_SHAPE = (240, 320, 3)
# Feature schema definition for the LeRobot dataset
FEATURES = {
    "head_cam": {"dtype": "video", "shape": CAMERA_SHAPE, "names": ["h", "w", "c"]},
    "endpose": {"dtype": "float32", "shape": (14,), "names": ["endpose"]},
    "agent_pos": {"dtype": "float32", "shape": (14,), "names": ["agent_pos"]},
    "action": {"dtype": "float32", "shape": (14,), "names": ["action"]},
}


def _initialize_vggt_encoder(vggt_path: str) -> VGGTEncoder:
    """Initialize and load the VGGT encoder model."""
    encoder = VGGTEncoder(intermediate_layer_idx=[4, 11, 17, 23])
    encoder.load_pretrained_model(vggt_path)
    encoder.to("cuda").eval()
    return encoder

def _extract_vggt_features(rgb: np.ndarray, encoder: VGGTEncoder) -> Dict[str, torch.Tensor]:
    """Extract VGGT features from RGB image."""
    # Convert to tensor and preprocess
    img_t = torch.from_numpy(rgb.transpose(2, 0, 1) / 255.0).to(dtype=torch.float32, device="cuda")
    img_t = preprocess_images(img_t.unsqueeze(0))  # (1, 3, H, W)
    
    # Extract features
    with torch.no_grad():
        vggt_features = encoder(img_t.unsqueeze(0))  # (1, 1, 3, H, W)
    
    torch.cuda.empty_cache()
    return vggt_features

def _extract_frame(data: Dict[str, Any]) -> Dict[str, Any]:
    """Convert raw pickle dictionary to the format expected by LeRobot."""
    return {
        "task": TASK_STR,
        "agent_pos": data["joint_action"].astype(np.float32),
        "action": data["joint_action"].astype(np.float32),
        "endpose": data["endpose"].astype(np.float32),
        "head_cam": data["observation"]["head_camera"]["rgb"].astype(np.float32) / 255.0,
    }

def _update_feature_buffer(buffer: Dict[str, List], features: Dict[str, torch.Tensor]) -> None:
    """Update VGGT feature buffer with new features."""
    feature_mappings = [
        ('spatial_tokens_4', features['spatial_tokens_list'][4]),
        ('spatial_tokens_11', features['spatial_tokens_list'][11]),
        ('spatial_tokens_17', features['spatial_tokens_list'][17]),
        ('spatial_tokens_23', features['spatial_tokens_list'][23]),
        ('camera_tokens_4', features['camera_tokens_list'][4]),
        ('camera_tokens_11', features['camera_tokens_list'][11]),
        ('camera_tokens_17', features['camera_tokens_list'][17]),
        ('camera_tokens_23', features['camera_tokens_list'][23]),
        ('image_tokens', features['image_tokens']),
        ('image_tokens_pos', features['image_tokens_pos']),
    ]
    
    for name, tensor in feature_mappings:
        buffer[name].append(tensor.squeeze().cpu().numpy())

def _save_features_to_disk(buffer: Dict[str, List], repo: str, ep_idx: int, dst_dir: str) -> None:
    """Save VGGT features buffer to disk as .npy files."""
    for key, value in buffer.items():
        output_dir = Path(f"{dst_dir}/npy/{key}")
        output_dir.mkdir(parents=True, exist_ok=True)
        np.save(output_dir / f"episode_{ep_idx:06d}.npy", value)

def _process_episode(dataset: LeRobotDataset, episode_dir: Path, encoder: VGGTEncoder):
    """Convert raw pickle dictionary to the format expected by LeRobot."""

    # Sort pickle files numerically to preserve temporal order
    pkl_files = sorted(episode_dir.glob("*.pkl"), key=lambda p: int(p.stem))

    # Initialize VGGT features buffer
    vggt_features_buffer = {
        'spatial_tokens_4': [], 'spatial_tokens_11': [], 'spatial_tokens_17': [], 'spatial_tokens_23': [],
        'camera_tokens_4': [], 'camera_tokens_11': [], 'camera_tokens_17': [], 'camera_tokens_23': [],
        'image_tokens': [], 'image_tokens_pos': [],
    }
    
    for pkl_path in tqdm(pkl_files, desc=f"{episode_dir.stem}:", leave=False):
        # Load pickle data
        with pkl_path.open("rb") as f:
            raw = pickle.load(f)
        
        frame = _extract_frame(raw)
        dataset.add_frame(frame)

        # Process image
        rgb = raw["observation"]["head_camera"]["rgb"]  # uint8 (H, W, 3)

        # Extract VGGT features
        vggt_features = _extract_vggt_features(rgb, encoder)

        # Update feature buffer
        _update_feature_buffer(vggt_features_buffer, vggt_features)
    
    # Save episode and features
    dataset.save_episode()
    _save_features_to_disk(vggt_features_buffer, args.repo, int(episode_dir.stem[len("episode"):]), args.dst_dir)
    

def main(args):
    # Initialize VGGT encoder
    encoder = _initialize_vggt_encoder(args.vggt_path)
    
    # Create LeRobot dataset
    ds = LeRobotDataset.create(
        root=args.dst_dir,
        repo_id=args.repo,
        fps=args.fps,
        robot_type="AgileBot",
        features=FEATURES,
        use_videos=True,
    )
    for episode_idx in range(len([d for d in os.listdir(args.src_dir) if os.path.isdir(os.path.join(args.src_dir, d))])):
        episode_dir = Path(os.path.join(args.src_dir, f"episode{episode_idx}"))

        assert os.path.isdir(episode_dir), f"Episode directory {episode_dir} does not exist"

        _process_episode(ds, episode_dir, encoder)
    
    # Push to hub if requested
    if args.push:
        ds.push_to_hub(push_videos=True, tags=["emvis", "converted"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_dir", type=str, required=True, help="Source directory containing episode folders, robotwin format")
    parser.add_argument("--dst_dir", type=str, required=True, help="Destination directory for the dataset")
    parser.add_argument("--repo", type=str, required=True, help="HuggingFace repository ID")
    parser.add_argument("--fps", type=int, default=40, help="Frames per second for the dataset")
    parser.add_argument("--push", type=bool, default=False, help="Whether to push the dataset to the HuggingFace Hub")
    parser.add_argument("--vggt_path", type=str, default="model.pt", help="Path to the VGGT model")
    args = parser.parse_args()

    if os.path.exists(args.dst_dir):
        shutil.rmtree(args.dst_dir)

    main(args)
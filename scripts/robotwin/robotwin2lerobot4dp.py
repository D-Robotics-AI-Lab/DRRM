import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any
import numpy as np
import argparse
from loguru import logger
import shutil

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

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


def _extract_frame(data: Dict[str, Any]) -> Dict[str, Any]:
    """Convert raw pickle dictionary to the format expected by LeRobot."""
    return {
        "task": TASK_STR,
        "agent_pos": data["joint_action"].astype(np.float32),
        "action": data["joint_action"].astype(np.float32),
        "endpose": data["endpose"].astype(np.float32),
        "head_cam": data["observation"]["head_camera"]["rgb"].astype(np.float32) / 255.0,
    }

def _process_episode(dataset: LeRobotDataset, episode_dir: Path) -> None:
    """Convert a single episode and append its frames to *ds*."""

    # Sort pickle files numerically to preserve temporal order
    pkl_files = sorted(episode_dir.glob("*.pkl"), key=lambda p: int(p.stem))

    for pkl_path in pkl_files:
        with pkl_path.open("rb") as f:
            raw = pickle.load(f)

        frame = _extract_frame(raw)
        dataset.add_frame(frame)

    # Persist episode to disk so that an unexpected crash does not lose work.
    dataset.save_episode()


def main(args):
    # Create (or open) the target dataset
    ds = LeRobotDataset.create(
        root=args.dst_dir,
        repo_id=args.repo,
        fps=args.fps,
        robot_type="AgileBot",
        features=FEATURES,
        image_writer_threads=4,  # Use async PNG writing to improve I/O performance
        use_videos=True,
    )
    for episode_idx in range(len([d for d in os.listdir(args.src_dir) if os.path.isdir(os.path.join(args.src_dir, d))])):
        episode_dir = Path(os.path.join(args.src_dir, f"episode{episode_idx}"))

        assert os.path.isdir(episode_dir), f"Episode directory {episode_dir} does not exist"

        _process_episode(ds, episode_dir)

    if args.push:
        logger.info("Pushing dataset to the HuggingFace Hub …")
        ds.push_to_hub(push_videos=True, tags=["emvis", "converted"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_dir", type=str, required=True, help="Source directory containing episode folders, robotwin format")
    parser.add_argument("--dst_dir", type=str, required=True, help="Destination directory for the dataset")
    parser.add_argument("--repo", type=str, required=True, help="HuggingFace repository ID")
    parser.add_argument("--fps", type=int, default=40, help="Frames per second for the dataset")
    parser.add_argument("--push", type=bool, default=False, help="Whether to push the dataset to the HuggingFace Hub")
    args = parser.parse_args()

    if os.path.exists(args.dst_dir):
        shutil.rmtree(args.dst_dir)

    main(args)
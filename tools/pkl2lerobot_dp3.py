#!/usr/bin/env python
"""
Convert *.pkl episodes to LeRobot v2.1 dataset format (videos separated).

Usage:
python scripts/pkl2lerobot_dp3.py \
      --root ./data/dual_bottles_pick_hard_D435_pc_pkl \
      --repo D-robotics/dual_bottles_pick_hard_D435 \
      --episodes 50 \
      --fps 40 
"""

from __future__ import annotations

import os
import sys
import pickle
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any

# Allow running the script directly without installing the whole project as a package
sys.path.append(str(Path(__file__).resolve().parent.parent))
# -----------------------------------------------------------------------------
# Environment defaults
# -----------------------------------------------------------------------------

# Respect user-defined values, otherwise create reasonable defaults that work in
# common Docker / remote scenarios.
os.environ["HF_HOME"] = "/workspace/.cache/huggingface"
os.environ["HF_LEROBOT_HOME"] = "/workspace/.cache/huggingface/lerobot"
import numpy as np
import tyro
from tqdm import tqdm
from drrm.dataset.drrm_dataset import DRRMDataset



# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Feature schema definition for the LeRobot dataset
FEATURES: Dict[str, Dict[str, Any]] = {
    # Point cloud data: 1024 points, each with XYZ coordinates and RGB colors
    "point_cloud": {"dtype": "float32", "shape": (1024, 6), "names": ["x", "y", "z", "r", "g", "b"]},
    
    # Robot state and action data
    "endpose": {"dtype": "float32", "shape": (14,), "names": ["endpose"]},
    "agent_pos": {"dtype": "float32", "shape": (14,), "names": ["agent_pos"]},
    "action": {"dtype": "float32", "shape": (14,), "names": ["action"]},
}

# Task name for dataset labeling (modify as needed)
TASK_STR: str = "dual bottles pick hard"

# -----------------------------------------------------------------------------
# CLI dataclass
# -----------------------------------------------------------------------------

@dataclass
class Args:
    """Command-line arguments for the conversion script."""

    root: Path  # Root directory containing episode data
    repo: str  # HuggingFace repository ID for uploading
    episodes: int = 100  # Number of episodes to process
    fps: int = 10  # Frames per second for the dataset
    push: bool = False  # Whether to push the dataset to HuggingFace

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def _extract_frame(data: Dict[str, Any]) -> Dict[str, Any]:
    """Convert raw pickle dictionary to the format expected by LeRobot."""
    return {
        "task": TASK_STR,
        "point_cloud": data["pointcloud"][:,:].astype(np.float32),
        "agent_pos": data["joint_action"].astype(np.float32),
        "action": data["joint_action"].astype(np.float32),
        "endpose": data["endpose"].astype(np.float32),
    }

def _process_episode(ds: DRRMDataset, ep_dir: Path, ep_idx: int) -> None:
    """Convert a single episode and append its frames to *ds*."""

    logger.info("Processing episode %d", ep_idx)

    # Sort pickle files numerically to preserve temporal order
    pkl_files = sorted(ep_dir.glob("*.pkl"), key=lambda p: int(p.stem))

    for pkl_path in tqdm(pkl_files, desc=f"Ep{ep_idx}", leave=False):
        with pkl_path.open("rb") as f:
            raw = pickle.load(f)

        frame = _extract_frame(raw)
        ds.add_frame(frame)

    # Persist episode to disk so that an unexpected crash does not lose work.
    ds.save_episode()

# -----------------------------------------------------------------------------
# Main entry point
# -----------------------------------------------------------------------------

def main(args: Args) -> None:  # noqa: D401  (simple function name is intentional)
    """Run conversion based on *args*."""

    # Create (or open) the target dataset
    ds = DRRMDataset.create(
        repo_id=args.repo,
        fps=args.fps,
        robot_type="AgileBot",
        features=FEATURES,
        image_writer_threads=4,  # Use async PNG writing to improve I/O performance
        use_videos=True,
    )

    for ep_idx in range(args.episodes):
        ep_dir = args.root / f"episode{ep_idx}"

        if not ep_dir.is_dir():
            logger.warning("Directory %s does not exist – stopping at episode %d", ep_dir, ep_idx)
            break

        _process_episode(ds, ep_dir, ep_idx)

    if args.push:
        logger.info("Pushing dataset to the HuggingFace Hub …")
        ds.push_to_hub(push_videos=True, tags=["emvis", "converted"])

if __name__ == "__main__":
    main(tyro.cli(Args))

import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import argparse
from loguru import logger
import shutil

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from tqdm import tqdm

# Constants
# TASK_STR = "dual bottles pick easy"
CAMERA_SHAPE = (240, 320, 3)
# Feature schema definition for the LeRobot dataset
FEATURES = {
    "head_cam": {"dtype": "image", "shape": CAMERA_SHAPE, "names": ["h", "w", "c"]},
    "front_cam": {"dtype": "image", "shape": CAMERA_SHAPE, "names": ["h", "w", "c"]},
    "endpose": {"dtype": "float32", "shape": (14,), "names": ["endpose"]},
    "agent_pos": {"dtype": "float32", "shape": (14,), "names": ["agent_pos"]},
    "action": {"dtype": "float32", "shape": (14,), "names": ["action"]}
}

TASK_DICT = {
    'block_hammer_beat': "Grab the hammer and beat the block.",
    'put_apple_cabinet': "Use one arm to open the cabinet's drawer, and use another arm to put the apple on the table to the drawer.",
    'bottle_adjust': "Pick up the bottle on the table headup with the correct arm.",
    'container_place': "Place the container onto the plate.",
    'block_handover': "Use the left arm to grasp the red block on the table, handover it to the right arm and place it on the blue pad.",
    'blocks_stack_easy': "Move the blocks to the center of the table, and stack the geen block on the red block.",
    'blocks_stack_hard': "Move the blocks to the center of the table, and stack the blue block on the green block, and the green block on the red block.",
    'diverse_bottles_pick': "Pick up one bottle with one arm, and pick up another bottle with the other arm.",
    'dual_bottles_pick_easy': "Pick up one bottle with one arm, and pick up another bottle with the other arm.",
    'dual_bottles_pick_hard': "Pick up one bottle with one arm, and pick up another bottle with the other arm.",
    'dual_shoes_place': "Use both arms to pick up the two shoes on the table and put them in the shoebox, with the shoe tip pointing to the left.",
    'empty_cup_place': "Use an arm to place the empty cup on the coaster.",
    'mug_hanging_easy': "Use left arm to pick the mug on the table, rotate the mug and put the mug down in the middle of the table, use the right arm to pick the mug and hang it onto the rack.",
    'mug_hanging_hard': "Use left arm to pick the mug on the table, rotate the mug and put the mug down in the middle of the table, use the right arm to pick the mug and hang it onto the rack.",
    'pick_apple_messy': "Pick apple",
    'shoe_place': "Use one arm to grab the shoe from the table and place it on the mat.",
    'tool_adjust': "Pick up the tool on the table headup with the correct arm.",
}

def _extract_frame(data: Dict[str, Any], task: str) -> Dict[str, Any]:
    """Convert raw pickle dictionary to the format expected by LeRobot."""
    return {
        # "instructions": [TASK_DICT[task]],
        "agent_pos": data["joint_action"].astype(np.float32),
        "action": data["joint_action"].astype(np.float32),
        "endpose": data["endpose"].astype(np.float32),
        "head_cam": data["observation"]["head_camera"]["rgb"].astype(np.float32) / 255.0,
        "front_cam": data["observation"]["front_camera"]["rgb"].astype(np.float32) / 255.0,
    }

def _process_episode(dataset: LeRobotDataset, episode_dir: Path, task: str) -> None:
    """Convert a single episode and append its frames to *ds*."""

    # Sort pickle files numerically to preserve temporal order
    pkl_files = sorted(episode_dir.glob("*.pkl"), key=lambda p: int(p.stem))

    for pkl_path in pkl_files:
        with pkl_path.open("rb") as f:
            raw = pickle.load(f)

        frame = _extract_frame(raw, task)
        dataset.add_frame(frame, task=task)

    # Persist episode to disk so that an unexpected crash does not lose work.
    dataset.save_episode()


def main(args):
    pt_str = "robotwin_v1_D435_pretrain"
    pt_dst = os.path.join(args.dst_dir, pt_str)
    # Create (or open) the target dataset
    if os.path.exists(pt_dst):
        shutil.rmtree(pt_dst)
    ds_pt = LeRobotDataset.create(
        root=pt_dst,
        repo_id=os.path.join(args.repo, pt_str),
        fps=args.fps,
        robot_type="AgilexBot",
        features=FEATURES,
        image_writer_threads=4,  # Use async PNG writing to improve I/O performance
        use_videos=False,
    )

    ft_str = "robotwin_v1_D435_finetuning"
    ft_dst = os.path.join(args.dst_dir, ft_str)
    if os.path.exists(ft_dst):
        shutil.rmtree(ft_dst)
    ds_ft = LeRobotDataset.create(
        root=ft_dst,
        repo_id=os.path.join(args.repo, ft_str),
        fps=args.fps,
        robot_type="AgilexBot",
        features=FEATURES,
        image_writer_threads=4,  # Use async PNG writing to improve I/O performance
        use_videos=False,
    )

    for task in TASK_DICT:
        src_dir = os.path.join(args.src_dir, f"{task}_D435_pkl_200")
        idx_len = len([d for d in os.listdir(src_dir) if os.path.isdir(os.path.join(src_dir, d))])

        ft_mask = np.zeros(idx_len, dtype=bool)
        rng = np.random.default_rng(seed=42)
        ft_idxs = rng.choice(idx_len, size=args.ft_num, replace=False)
        ft_mask[ft_idxs] = True

        for episode_idx, ft_ep in enumerate(tqdm(ft_mask)):
            episode_dir = Path(os.path.join(src_dir, f"episode{episode_idx}"))
            assert os.path.isdir(episode_dir), f"Episode directory {episode_dir} does not exist"

            if ft_ep:
                _process_episode(ds_ft, episode_dir, task)
            else:
                _process_episode(ds_pt, episode_dir, task)

    if args.push:
        logger.info("Pushing dataset to the HuggingFace Hub …")
        ds_pt.push_to_hub(push_videos=True, tags=["emvis", "converted"])
        ds_ft.push_to_hub(push_videos=True, tags=["emvis", "converted"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_dir", type=str, required=True, help="Source directory containing episode folders, robotwin format")
    parser.add_argument("--dst_dir", type=str, required=True, help="Destination directory for the dataset")
    parser.add_argument("--repo", type=str, required=True, help="HuggingFace repository ID")
    parser.add_argument("--ft_num", type=int, default=100, help="ft num")
    # parser.add_argument("--task", type=str, required=True, help="task name")
    parser.add_argument("--fps", type=int, default=40, help="Frames per second for the dataset")
    parser.add_argument("--push", type=bool, default=False, help="Whether to push the dataset to the HuggingFace Hub")
    args = parser.parse_args()

    # global TASK_STR
    # TASK_STR = ' '.join(args.task.split('_'))

    main(args)
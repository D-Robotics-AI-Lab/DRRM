import time

import accelerate
from torch.utils.data import DataLoader, RandomSampler, BatchSampler
from tqdm import tqdm

import sys
from pathlib import Path
import os
# Add the parent directory to Python path to allow importing from dataset module
sys.path.append(str(Path(__file__).parent.parent))
os.environ["HF_HOME"] = "/workspace/.cache/huggingface"
os.environ["HF_LEROBOT_HOME"] = "/workspace/.cache/huggingface/lerobot"
from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
from drrm.datasets.drrm_dataset import DRRMDataset
# -----------------------------
# Configuration
# -----------------------------
HORIZON = 8
N_OBS_STEPS = 3
N_ACTION_STEPS = 8
BATCH_SIZE = 128
EPOCH_MULTIPLIER = 300      # Number of batches relative to dataset length
VAL_RATIO = 0.02
SEED = 45

USE_VGGT_INPUT = True
VGGT_FEATURE_KEYS = [
    "spatial_tokens_23",
    "image_tokens_pos",
]

REPO_ID = "D-robotics/dual_bottles_pick_hard_D435_emvis"

# -----------------------------
# Dataset & Dataloader
# -----------------------------
metadata = LeRobotDatasetMetadata(repo_id=REPO_ID)

dataset = DRRMDataset(
    repo_id=REPO_ID,
    horizon=HORIZON,
    pad_before=N_OBS_STEPS - 1,
    pad_after=N_ACTION_STEPS - 1,
    seed=SEED,
    val_ratio=VAL_RATIO,
    max_train_episodes=None,
    npy_feature_keys=VGGT_FEATURE_KEYS,
    dataset_metadata=metadata,
)

normalizer = dataset.get_normalizer()

rand_sampler = RandomSampler(
    dataset,
    replacement=True,
    num_samples=len(dataset) * EPOCH_MULTIPLIER,
)

batch_sampler = BatchSampler(
    rand_sampler,
    batch_size=BATCH_SIZE,
    drop_last=True,
)

dataloader = DataLoader(
    dataset,
    batch_sampler=batch_sampler,
    num_workers=4,
    prefetch_factor=1,
    pin_memory=False,
    persistent_workers=False,
)

# -----------------------------
# Acceleration & Benchmarking
# -----------------------------
accelerator = accelerate.Accelerator()
dataloader = accelerator.prepare(dataloader)

progress = tqdm if accelerator.is_main_process else lambda x: x

start_time = time.time()
for _ in progress(dataloader):
    pass

elapsed = time.time() - start_time

if accelerator.is_main_process:
    print(
        f"Total time elapsed: {elapsed:.2f}s "
        f"({elapsed / len(dataloader):.2f}s per batch)"
    )
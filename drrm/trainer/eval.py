import copy
import logging
import os
from pathlib import Path
import hydra
from omegaconf import OmegaConf

import diffusers
import torch
from torch.utils.data import RandomSampler, BatchSampler
import transformers
from transformers import AutoConfig, AutoModel
from transformers.models.deit.image_processing_deit import valid_images
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers.optimization import get_scheduler
from diffusers.utils import is_wandb_available
from tqdm.auto import tqdm

from drrm.models.base_runner import load_policy
from drrm.common.ema_model import EMAModel


if is_wandb_available():
    import wandb

def visualize_flow(val_dataloader, policy_model):
    # Only show the progress bar once on each machine.
    # progress_bar = tqdm(range(0, len(val_dataset)))
    # progress_bar.set_description("Steps")
    # progress_bar.update(0)

    
    policy_model.eval()
    val_losses = {}
    # Forward and backward...
    for step, batch in enumerate(val_dataloader):
        loss = policy_model(batch)
        torch.cuda.empty_cache()
        # progress_bar.update(1)

def eval(args, logger):
    # task_name, expert_data_num, ckpt_setting, checkpoint_num
    if args.checkpoint_num is None:
        checkpoint_dir = f"checkpoints/vodpp_{args.task_name}_{args.expert_data_num}_{args.ckpt_setting}"
    else:
        checkpoint_dir = f"checkpoints/vodpp_{args.task_name}_{args.expert_data_num}_{args.ckpt_setting}/checkpoint-{args.checkpoint_num}"
    policy_model = load_policy(checkpoint_dir, use_ckp_code=False, device='cuda')
    
    # Dataset and DataLoaders creation
    train_dataset = hydra.utils.instantiate(args.train_dataset)
    val_dataset = train_dataset.get_validation_dataset()
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
        persistent_workers=True,
    )

    

    return dict(loss_for_log)
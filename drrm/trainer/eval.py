import copy
import logging
import os
from pathlib import Path
import random
import hydra
import numpy as np
from omegaconf import OmegaConf

import diffusers
import torch
from torch.utils.data import RandomSampler, BatchSampler
from tqdm import tqdm
import transformers
from transformers import AutoConfig, AutoModel
from transformers.models.deit.image_processing_deit import valid_images
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers.optimization import get_scheduler
from diffusers.utils import is_wandb_available


from drrm.models.base_runner import load_policy
from drrm.common.ema_model import EMAModel

def set_seed(seed):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

if is_wandb_available():
    import wandb

def visualize_flow(val_dataloader, policy, prefix):
    # Only show the progress bar once on each machine.
    # progress_bar = tqdm(range(0, len(val_dataset)))
    # progress_bar.set_description("Steps")
    # progress_bar.update(0)
    n_obs_steps = policy.n_obs_steps
    policy.eval()
    # Forward and backward...
    with torch.no_grad():
        with torch.autocast(device_type=str(policy.device), dtype=torch.bfloat16):
            episode = []
            gt_episode = []
            for step, batch in enumerate(tqdm(val_dataloader)):
                # if step == 2: break
                frame_index = batch['obs'].pop('frame_index')
                episode_index = batch['obs'].pop('episode_index')
                task_index = batch['obs'].pop('task_index')
                # repeat each tensor along dimension 0 by 32 times
                sample_batch = {k: v.repeat(32, *([1] * (v.dim() - 1))) for k, v in batch['obs'].items()}
                flows = []
                # loss = policy(batch)
                for _ in range(32):
                    result = policy.predict_action(sample_batch, return_flow=True)
                    flow = result['flow']
                    flows.append(np.stack(flow, axis=1))
                # progress_bar.update(1)
                flows = np.concatenate(flows, axis=0)
                episode.append(flows)
                gt_episode.append(result['gt'].cpu().numpy()[0,-1,:])
            torch.cuda.empty_cache()
                
    episode = np.stack(episode, axis=0)
    gt_episode = np.stack(gt_episode, axis=0)
    # episode = episode.reshape(*(episode.shape[:3]), -1)  # (num_segment, num_flows, num_step, horizon * 14)
    episode = episode[..., -1, :]  # (num_segment, num_flows, num_step, horizon * 14)
    # Fit KernelPCA on features at time index 0, then transform other time steps
    from sklearn.decomposition import KernelPCA
    kpca = KernelPCA(n_components=1, kernel='rbf', fit_inverse_transform=False, random_state=0)
    num_segment, num_flows, num_step, feat_dim = episode.shape
    # Fit KPCA on episode[:, :, 0, :] (all segments and flows at t=0)
    t0 = episode[:, :, 0, :].reshape(-1, feat_dim)  # (num_segment * num_flows, feat_dim)
    kpca.fit(t0)
    # Transform each time step using the fitted KPCA
    episode_pca = np.zeros((num_segment, num_flows, num_step), dtype=np.float32)
    for ti in range(num_step):
        Xi = episode[:, :, ti, :].reshape(-1, feat_dim)
        transformed = kpca.transform(Xi)  # shape (num_segment * num_flows, 1)
        if ti == num_step-1:
            gt_pca = kpca.transform(gt_episode)
        episode_pca[:, :, ti] = transformed.reshape(num_segment, num_flows)

    # Normalize flows and gt for visualization
    norm_flow = np.zeros((num_segment, num_flows, num_step), dtype=np.float32)
    norm_gt = np.zeros((num_segment), dtype=np.float32)
    for i in range(num_segment):
        for j in range(num_step):
            mean = episode_pca[i, :, j].mean()
            std = episode_pca[i, :, j].std()
            norm_flow[i, :, j] = (episode_pca[i, :, j] - mean) / (std+1e-8)
        norm_gt[i] = (gt_pca[i] - mean) / (std+1e-8)
    # Also compute global normalization
    mean = episode_pca[:, :, 0].mean()
    std = episode_pca[:, :, 0].std()
    global_flow = (episode_pca - mean) / (std+1e-8)
    global_gt = (gt_pca - mean) / (std+1e-8)

    import matplotlib.pyplot as plt
    out_dir = './outputs/flows'
    os.makedirs(out_dir, exist_ok=True)
    time_steps = np.arange(num_step)
    fig, axes = plt.subplots(2, 1, figsize=((num_step-1) * num_segment, 14))
    # axes = axes[0]
    for l, (flow, gt) in enumerate([(norm_flow, norm_gt), (global_flow, global_gt)]):
        ax = axes[l]
        for i in range(num_segment):
            time_steps = np.arange(num_step)+(i*num_step)
            for f in range(num_flows):
                ax.plot(time_steps, flow[i, f, :], label=f'flow_{f}', color='tab:blue', linewidth=10, alpha=0.015)
            ax.plot(time_steps, flow[i, f, :], color='tab:green', label=f'flow_{f}', linewidth=5, zorder=99)
            ax.plot((np.arange(num_segment)+1)*num_step-1, gt, color='tab:red', label='gt', linewidth=2, zorder=100)

        ax.set_xlabel('Time Step')
        ax.set_ylabel('PCA Flow Value')
        ax.set_title(f'Normalized Flows' if l == 0 else 'Global Flows')
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.set_ylim(-3.5, 3.5)
        if num_flows <= 20:
            ax.legend(loc='upper right', fontsize='x-small', ncol=1)
    
    plt.tight_layout()
    out_path = os.path.join(out_dir, f'{prefix}_flow_visualization.png')
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print("Flow visualizations saved.")


def eval(args, logger):
    # task_name, expert_data_num, ckpt_setting, checkpoint_num
    set_seed(args.seed)
    prefix = args.ckpt_setting
    if args.get('checkpoint_num', None) is None:
        checkpoint_dir = f"checkpoints/vodpp_{args.task_name}_{args.expert_data_num}_{args.ckpt_setting}"
    else:
        checkpoint_dir = f"checkpoints/vodpp_{args.task_name}_{args.expert_data_num}_{args.ckpt_setting}/checkpoint-{args.checkpoint_num}"
        prefix = f"{prefix}_{args.checkpoint_num}"
    policy_model = load_policy(checkpoint_dir, use_ckp_code=False, device='cuda')
    horizon = policy_model.horizon
    n_obs_steps = policy_model.n_obs_steps
    n_act_steps = horizon - n_obs_steps
    
    # Dataset and DataLoaders creation
    train_dataset = hydra.utils.instantiate(args.train_dataset)
    val_dataset = train_dataset.get_validation_dataset()
    # 实现batch_sampler要求顺序采样，batch size为1，stride为n_act_steps
    class SequentialStrideSampler(torch.utils.data.Sampler):
        """Yield indices 0, stride, 2*stride, ... for sequential sampling."""
        def __init__(self, data_source, stride=1):
            self.data_source = data_source
            self.stride = int(stride)

        def __iter__(self):
            n = len(self.data_source)
            return iter(range(0, n, self.stride))

        def __len__(self):
            n = len(self.data_source)
            return (n + self.stride - 1) // self.stride

    seq_sampler = SequentialStrideSampler(val_dataset, stride=n_act_steps)
    batch_sampler = BatchSampler(seq_sampler, batch_size=1, drop_last=False)

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_sampler=batch_sampler,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
        persistent_workers=True,
    )

    visualize_flow(val_dataloader, policy_model, prefix)

    # return dict(loss_for_log)
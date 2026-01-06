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
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize
from matplotlib import cm

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
        
def set_seed(seed):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

if is_wandb_available():
    import wandb

def pca(episode, *addition):
    from sklearn.decomposition import KernelPCA
    kpca = KernelPCA(n_components=1, kernel='rbf', fit_inverse_transform=False, random_state=0)
    num_segment, num_flows, num_step, feat_dim = episode.shape
    # Fit KPCA on episode[:, :, 0, :] (all segments and flows at t=0)
    t0 = episode[:, :, 0, :].reshape(-1, feat_dim)  # (num_segment * num_flows, feat_dim)
    kpca.fit(t0)
    # Transform each time step using the fitted KPCA
    episode_pca = np.zeros((num_segment, num_flows, num_step), dtype=np.float32)
    addition_pca = []
    for ti in range(num_step):
        Xi = episode[:, :, ti, :].reshape(-1, feat_dim)
        transformed = kpca.transform(Xi)  # shape (num_segment * num_flows, 1)
        if ti == num_step-1:
            for add in addition:
                addition_pca.append(kpca.transform(add))
        episode_pca[:, :, ti] = transformed.reshape(num_segment, num_flows)
    return episode_pca, addition_pca

def tnse(episode, *addition):
    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=1, random_state=0, perplexity=min(30, episode.shape[0]-1))
    all_data = np.concatenate([episode, *addition], axis=0)
    all_embedded = tsne.fit_transform(all_data)
    
    episode_pca = all_embedded[:episode.shape[0]]
    addition_pca = []
    if addition:
        addition_embedded = all_embedded[episode.shape[0]:]
        ind = 0
        for ep in addition:
            addition_pca.append(addition_embedded[ind:ind+ep.shape[0]])
            ind += ep.shape[0]
    return episode_pca, addition_pca

def plot_gradient_line(ax, x, y, cmap='viridis', linewidth=2.0, zorder=100, label=None):
    """Plot a 2D line with a color gradient along its length.

    Uses a LineCollection with per-segment colors. Also adds an
    invisible proxy line for legend entries when `label` is provided.
    """
    x = np.asarray(x).reshape(-1)
    y = np.asarray(y).reshape(-1)
    if x.size < 2 or y.size < 2:
        return

    points = np.stack([x, y], axis=1).reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    lc = LineCollection(segments, cmap=cm.get_cmap(cmap), norm=Normalize(0.0, 1.0))
    lc.set_array(np.linspace(0.0, 1.0, len(segments)))
    lc.set_linewidth(linewidth)
    lc.set_zorder(zorder)
    ax.add_collection(lc)
    ax.autoscale_view()

    if label is not None:
        # Proxy for legend
        ax.plot([], [], color=cm.get_cmap(cmap)(0.8), linewidth=linewidth, label=label)

def visualize_traj(val_dataset, policy, prefix):
    horizon = policy.horizon
    n_obs_steps = policy.n_obs_steps
    n_act_steps = horizon - n_obs_steps
    policy.eval()

    seq_sampler = SequentialStrideSampler(val_dataset, stride=1)
    batch_sampler = BatchSampler(seq_sampler, batch_size=1, drop_last=False)
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_sampler=batch_sampler,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )
    
    # Forward and backward...
    with torch.no_grad():
        with torch.autocast(device_type=str(policy.device), dtype=torch.bfloat16):
            episode = []
            gt_episode = []
            obs_episode = []
            img_episode = []
            state_episode = []
            for step, batch in enumerate(tqdm(val_dataloader)):
                # if step == 6: break
                sample_batch = {k: v.repeat(32, *([1] * (v.dim() - 1))) for k, v in batch['obs'].items()}
                flows = []
                # for _ in range(4):
                result = policy.predict_action(sample_batch, return_flow=True)
                flow = result['flow']
                flows.append(np.stack(flow, axis=1))

                flows = np.concatenate(flows, axis=0)
                episode.append(flows)
                gt_episode.append(result['gt'].cpu().numpy()[0,-1,:])
                state_episode.append(result['state'].cpu().numpy()[0,0,:])
                obs_episode.append(result['obs'].to(device='cpu', dtype=torch.float32).numpy()[0])
                img_episode.append(result['img'].to(device='cpu', dtype=torch.float32).numpy()[0])
            torch.cuda.empty_cache()
                
    episode = np.stack(episode, axis=0)
    episode = episode[..., -1, :]  # (num_segment, num_flows, num_step, horizon * 14)
    gt_episode = np.stack(gt_episode, axis=0)
    state_episode = np.stack(state_episode, axis=0)
    obs_episode = np.stack(obs_episode, axis=0)
    obs_episode = obs_episode.reshape(obs_episode.shape[0], -1)
    img_episode = np.stack(img_episode, axis=0).transpose(0, 2, 3, 1)
    img_episode = img_episode.reshape(img_episode.shape[0], -1)
    num_segment, num_flows, num_step, feat_dim = episode.shape
    
    # episode_pca, (gt_pca, state_pca) = tnse(episode.reshape(-1, feat_dim), gt_episode, state_episode)
    # episode_pca = episode_pca.reshape(num_segment, num_flows, num_step)
    episode_pca, (gt_pca, state_pca) = pca(episode, gt_episode, state_episode)
    
    # Use KernelPCA to transform obs_episode
    obs_pca, _ = tnse(obs_episode)
    norm_obs = (obs_pca - obs_pca.mean()) / (obs_pca.std() + 1e-8)
    img_pca, _ = tnse(img_episode)
    norm_img = (img_pca - img_pca.mean()) / (img_pca.std() + 1e-8)

    # Normalize flows and gt for visualization
    norm_flow = np.zeros((num_segment, num_flows, num_step), dtype=np.float32)
    for j in range(num_step):
        mean = episode_pca[:, :, j].mean()
        std = episode_pca[:, :, j].std()
        norm_flow[:, :, j] = (episode_pca[:, :, j] - mean) / (std+1e-8)
    norm_gt = (gt_pca - mean) / (std+1e-8)
    norm_state = (state_pca - mean) / (std+1e-8)

    import matplotlib.pyplot as plt
    out_dir = './outputs/traj'
    os.makedirs(out_dir, exist_ok=True)
    fig, axes = plt.subplots(
        4, 4, figsize=(4 * 25, 16),
        gridspec_kw={'height_ratios': [4.0, 0.3, 4.0, 0.3]},
    )

    def draw_subplot(axes, x, y, norm_gt, title, xlabel):
        ax = axes[0]
        sort_idx = np.argsort(x, axis=0)
        for f in range(y.shape[1]):
            ax.scatter(x, y[:, f], label=f'flow_{f}', color='tab:blue', linewidth=0.1, alpha=(1./y.shape[1]))
        plot_gradient_line(ax, x[sort_idx], y[sort_idx, f], cmap='autumn', linewidth=2, zorder=100, label=f'traj_{f}')
        ax.set_xlabel(xlabel)
        ax.set_ylabel('PCA State')
        ax.set_title(title)
        ax.grid(True, linestyle='--', alpha=0.3)

        ax = axes[1]
        # Draw a colorbar-like density strip showing distribution of x
        # Compute histogram and normalize to [0,1]
        x_hist, x_edges = np.histogram(x, bins=50)
        x_hist = x_hist.astype(np.float32)
        # Interpolate histogram to get smoother color transitions
        centers = 0.5 * (x_edges[1:] + x_edges[:-1])
        upsample_x = np.linspace(x_edges[0], x_edges[-1], len(x_hist) * 4)
        x_hist = np.interp(upsample_x, centers, x_hist)
        if x_hist.max() > 0:
            x_hist = (x_hist - x_hist.min()) / (x_hist.max() - x_hist.min() + 1e-8)
        else:
            x_hist[:] = 0.0
        # Render as a horizontal strip using imshow
        im = ax.imshow(
            x_hist[np.newaxis, :],
            cmap='viridis',
            aspect='auto',
            extent=[x_edges[0], x_edges[-1], 0.0, 1.0],
            interpolation='nearest',
        )
        ax.set_yticks([])
        ax.set_xticks([])
        ax.grid(False)

    draw_subplot(axes[:2,0], np.arange(num_segment), norm_flow[..., -1], norm_gt, 'Trajectory', 'Time')
    draw_subplot(axes[:2,1], norm_state[..., 0], norm_flow[..., -1], norm_gt, 'State-Prediction', 'State')
    draw_subplot(axes[:2,2], norm_img[..., 0], norm_flow[..., -1], norm_gt, 'Image-Prediction', 'Image')
    draw_subplot(axes[:2,3], norm_obs[..., 0], norm_flow[..., -1], norm_gt, 'Feature-Prediction', 'Feature')

    draw_subplot(axes[2:,0], np.arange(num_segment), norm_gt, norm_gt, 'Trajectory', 'Time')
    draw_subplot(axes[2:,1], norm_state[..., 0], norm_gt, norm_gt, 'State-GT', 'State')
    draw_subplot(axes[2:,2], norm_img[..., 0], norm_gt, norm_gt, 'Image-GT', 'Image')
    draw_subplot(axes[2:,3], norm_obs[..., 0], norm_gt, norm_gt, 'Feature-GT', 'Feature')
    
    plt.tight_layout()
    out_path = os.path.join(out_dir, f'{prefix}_traj_visualization.png')
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print("Flow visualizations saved.")


def visualize_flow(val_dataset, policy, prefix):
    horizon = policy.horizon
    n_obs_steps = policy.n_obs_steps
    n_act_steps = horizon - n_obs_steps
    policy.eval()

    seq_sampler = SequentialStrideSampler(val_dataset, stride=50)
    batch_sampler = BatchSampler(seq_sampler, batch_size=1, drop_last=False)
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_sampler=batch_sampler,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )
    
    # Forward and backward...
    with torch.no_grad():
        with torch.autocast(device_type=str(policy.device), dtype=torch.bfloat16):
            episode = []
            gt_episode = []
            for step, batch in enumerate(tqdm(val_dataloader)):
                # if step == 2: break
                # frame_index = batch['obs'].pop('frame_index')
                # episode_index = batch['obs'].pop('episode_index')
                # task_index = batch['obs'].pop('task_index')
                # repeat each tensor along dimension 0 by 32 times
                sample_batch = {k: v.repeat(32, *([1] * (v.dim() - 1))) for k, v in batch['obs'].items()}
                flows = []
                # loss = policy(batch)
                for _ in range(32):
                    result = policy.predict_action(sample_batch, return_flow=True)
                    flow = result['flow']
                    flows.append(np.stack(flow, axis=1))
                # progress_bar.update(1)
                flows = np.concatenate(flows, axis=0)[..., -1, :]  # (num_flows, num_step, horizon * 14)
                episode.append(flows)
                gt_episode.append(result['gt'].cpu().numpy()[0,-1,:])
            torch.cuda.empty_cache()
                
    episode = np.stack(episode, axis=0) # (num_segment, num_flows, num_step, horizon * 14)
    gt_episode = np.stack(gt_episode, axis=0)
    # episode = episode.reshape(*(episode.shape[:3]), -1)  # (num_segment, num_flows, num_step, horizon * 14)
    obs_episode = obs_episode.repeat(1, episode.shape[1], episode.shape[2], 1)
    num_segment, num_flows, num_step, feat_dim = episode.shape
    episode_pca, gt_pca = pca(episode, gt_episode)

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


def visualize_condition_flow(val_dataset, policy, prefix):
    horizon = policy.horizon
    n_obs_steps = policy.n_obs_steps
    n_act_steps = horizon - n_obs_steps
    policy.eval()

    seq_sampler = SequentialStrideSampler(val_dataset, stride=1)
    batch_sampler = BatchSampler(seq_sampler, batch_size=1, drop_last=False)
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_sampler=batch_sampler,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )
    
    # Forward and backward...
    with torch.no_grad():
        with torch.autocast(device_type=str(policy.device), dtype=torch.bfloat16):
            episode = []
            gt_episode = []
            obs_episode = []
            for step, batch in enumerate(tqdm(val_dataloader)):
                # if step == 2: break
                # frame_index = batch['obs'].pop('frame_index')
                # episode_index = batch['obs'].pop('episode_index')
                # task_index = batch['obs'].pop('task_index')
                # repeat each tensor along dimension 0 by 32 times
                sample_batch = {k: v.repeat(32, *([1] * (v.dim() - 1))) for k, v in batch['obs'].items()}
                flows = []
                # loss = policy(batch)
                for _ in range(32):
                    result = policy.predict_action(sample_batch, return_flow=True)
                    flow = result['flow']
                    flows.append(np.stack(flow, axis=1))
                # progress_bar.update(1)
                flows = np.concatenate(flows, axis=0)[..., -1, :]  # (num_flows, num_step, horizon * 14)
                obs = result['obs'].to(device='cpu', dtype=torch.float32).numpy()[0].reshape(-1) # (1, 1, 4 * 1024)
                # concat flows and obs along last dimension
                # cat_flows = np.concatenate([flows, np.tile(obs, (flows.shape[0], flows.shape[1], 1))], axis=-1)
                obs_episode.append(obs)
                episode.append(flows)
                gt_episode.append(result['gt'].cpu().numpy()[0,-1,:])
            torch.cuda.empty_cache()
                
    episode = np.stack(episode, axis=0) # (num_segment, num_flows, num_step, horizon * 14)
    num_segment, num_flows, num_step, feat_dim = episode.shape
    gt_episode = np.stack(gt_episode, axis=0)
    action_pca, (gt_pca,) = pca(episode, gt_episode)
    obs_episode = np.stack(obs_episode, axis=0)
    cat_episode = np.concatenate([np.tile(obs_episode[:,None,None,:], (1, num_flows, num_step, 1)), episode], axis=-1)
    # cat_episode = cat_episode.reshape(-1, cat_episode.shape[-1])
    # cat_pca, _ = tnse(cat_episode)
    # cat_pca = cat_pca.reshape(num_segment, num_flows, num_step)
    cat_pca = pca(cat_episode)[0]
    

    # Normalize flows and gt for visualization
    norm_action = np.zeros((num_segment, num_flows, num_step), dtype=np.float32)
    norm_gt = np.zeros((num_segment), dtype=np.float32)
    for i in range(num_segment):
        for j in range(num_step):
            mean = action_pca[i, :, j].mean()
            std = action_pca[i, :, j].std()
            norm_action[i, :, j] = (action_pca[i, :, j] - mean) / (std+1e-8)
        norm_gt[i] = (gt_pca[i] - mean) / (std+1e-8)
    # Also compute global normalization
    mean = action_pca[:, :, 0].mean()
    std = action_pca[:, :, 0].std()
    global_action = (action_pca - mean) / (std+1e-8)
    global_gt = (gt_pca - mean) / (std+1e-8)

    mean = cat_pca[:, :, 0].mean()
    std = cat_pca[:, :, 0].std()
    global_flow = (cat_pca - mean) / (std+1e-8)

    import matplotlib.pyplot as plt
    out_dir = './outputs/flows'
    os.makedirs(out_dir, exist_ok=True)
    time_steps = np.arange(num_step)
    fig, axes = plt.subplots(1, 1, figsize=(25, 25))
    # axes = axes[0]
    for l, (flow, action) in enumerate([(global_flow, norm_action)]):
        flow = np.concatenate([flow, action[...,-1:]], axis=-1)
        ax = axes
        for i in range(num_segment):
            # time_steps = np.arange(num_step)+(i*num_step)
            time_steps = np.arange(flow.shape[-1])
            for f in range(num_flows):
                ax.plot(time_steps, flow[i, f, :], label=f'flow_{f}', color='tab:blue', linewidth=5, alpha=0.005)
            ax.plot(time_steps, flow[i, f, :], color='tab:green', label=f'flow_{f}', linewidth=1, zorder=99)
            # ax.plot((np.arange(num_segment)+1)*num_step-1, gt, color='tab:red', label='gt', linewidth=2, zorder=100)

        ax.set_xlabel('Time Step')
        ax.set_ylabel('PCA Flow Value')
        ax.set_title(f'Normalized Flows' if l == 0 else 'Global Flows')
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.set_ylim(-2.5, 2.5)
        if num_flows <= 20:
            ax.legend(loc='upper right', fontsize='x-small', ncol=1)
    
    plt.tight_layout()
    out_path = os.path.join(out_dir, f'{prefix}_condition_flow_visualization.png')
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
        prefix = f"{prefix}_ckp{args.checkpoint_num}"
    policy_model = load_policy(checkpoint_dir, use_ckp_code=False, device='cuda')
    
    
    # Dataset and DataLoaders creation
    train_dataset = hydra.utils.instantiate(args.train_dataset)
    val_dataset = train_dataset.get_validation_dataset()
    # 实现batch_sampler要求顺序采样，batch size为1，stride为n_act_steps

    # visualize_flow(val_dataset, policy_model, prefix)
    visualize_condition_flow(val_dataset, policy_model, prefix)
    # visualize_traj(val_dataset, policy_model, prefix)

    # return dict(loss_for_log)
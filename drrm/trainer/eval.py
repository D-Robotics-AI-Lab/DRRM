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

def save_numpy(filename, **kwargs):
    out_dir = './outputs/npy'
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'{filename}.npy')
    np.save(out_path, kwargs)
    print("Flow numpy saved.")

def load_numpy(filename):
    out_dir = './outputs/npy'
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'{filename}.npy')
    data = np.load(out_path, allow_pickle=True).item()
    print("Flow numpy loaded.")
    return data

def pca(episode, *addition, n_components=1):
    from sklearn.decomposition import KernelPCA
    kpca = KernelPCA(n_components=n_components, kernel='rbf', fit_inverse_transform=False, random_state=0)
    num_segment, num_flows, num_step, feat_dim = episode.shape
    t0 = episode[:, :, 0, :].reshape(-1, feat_dim)  # (num_segment * num_flows, feat_dim)
    kpca.fit(t0)
    episode = episode[...,None,:].transpose(2, 0, 1, 3, 4)
    step_num = len(episode)
    addition = [*episode, *addition]

    addition_pca = []
    for add in addition:
        num_segment, num_flows, num_step, feat_dim = add.shape
        Xi = add.reshape(-1, feat_dim)
        transformed = kpca.transform(Xi)
        addition_pca.append(transformed.reshape(num_segment, num_flows, num_step, n_components))
    
    episode_pca = np.concatenate(addition_pca[:step_num], axis=-2)
    return episode_pca, addition_pca[step_num:]

def tnse(episode, *addition, n_components=1):
    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=n_components, random_state=0, perplexity=min(30, episode.shape[0]-1))
    all_data = []
    for data in [episode, *addition]:
        feat_dim = data.shape[-1]
        Xi = data.reshape(-1, feat_dim)
        all_data.append(Xi)
    all_data = np.concatenate(all_data, axis=0)
    all_embedded = tsne.fit_transform(all_data)
    
    all_data_pca = []
    for data in [episode, *addition]:
        item_num = np.prod(data.shape[:-1])
        embedded = all_embedded[:item_num, :]
        all_embedded = all_embedded[item_num:, :]
        all_data_pca.append(embedded.reshape(*data.shape[:-1], n_components))
        
    return all_data_pca[0], all_data_pca[1:]

def standardize(x):
    mean = x.mean()
    std = x.std()
    return (x - mean) / (std + 1e-8)

def min_max_normalize(x, y=None, shift=2.5, scale=5.0):
    if y is None: y = np.array(x.mean())
    min_ = min(x.min(), y.min())
    max_ = max(x.max(), y.max())

    x = (x - min_) / (max_ - min_)
    x = x * scale - shift

    y = (y - min_) / (max_ - min_)
    y = y * scale - shift
    return x, y

def align_data(x, y):
    from scipy.optimize import minimize
    def objective(params):
        """
        目标函数：计算 (m + X)*s - Y 的L2范数平方
        params: 包含[m, s]的数组
        """
        m, s = params
        residual = s * (m + x) - y  # 残差向量
        l2_norm_square = np.sum(residual ** 2)  # L2范数的平方
        return l2_norm_square

    # ---------------------- 3. 初始化参数并求解 ----------------------
    initial_guess = [0.0, 1.0]  # m和s的初始猜测值（可任意选，不影响最终结果）
    result = minimize(objective, initial_guess, method='L-BFGS-B')  # 用L-BFGS-B优化器

    return (result.x[0] + x) * result.x[1], y

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

def visualize_condition_sample_flow(val_dataset, policy, prefix):
    horizon = policy.horizon
    n_obs_steps = policy.n_obs_steps
    n_act_steps = horizon - n_obs_steps
    policy.eval()

    def get_pred_gt_data():
        val_dataset.detail_item = True
        seq_sampler = SequentialStrideSampler(val_dataset, stride=7)
        batch_sampler = BatchSampler(seq_sampler, batch_size=1, drop_last=False)
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset,
            batch_sampler=batch_sampler,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
        )
        
        with torch.no_grad():
            with torch.autocast(device_type=str(policy.device), dtype=torch.bfloat16):
                episode = []
                gt_episode = []
                obs_episode = []
                img_episode = []
                ind_list = []
                for step, batch in enumerate(tqdm(val_dataloader)):
                    frame_index = batch['obs'].pop('frame_index')
                    episode_index = batch['obs'].pop('episode_index')
                    task_index = batch['obs'].pop('task_index')
                    sample_batch = {k: v.repeat(4, *([1] * (v.dim() - 1))) for k, v in batch['obs'].items()}
                    flows = []
                    for _ in range(1):
                        result = policy.predict_action(sample_batch, return_flow=True)
                        flow = result['flow']
                        flows.append(np.stack(flow, axis=1))
                    flows = np.concatenate(flows, axis=0)
                    flows = flows.reshape(*flows.shape[:-2], -1)  # (num_flows, num_step, horizon * 14)
                    episode.append(flows)
                    obs = result['obs'].to(device='cpu', dtype=torch.float32).numpy()[0].reshape(-1) # (1, 1, 4 * 1024)
                    obs_episode.append(obs)
                    img = result['img'].to(device='cpu', dtype=torch.float32).numpy()[0].reshape(-1)
                    img_episode.append(img)
                    gt = result['gt'].cpu().numpy()[0].reshape(-1)
                    gt_episode.append(gt)
                    ind = torch.stack([episode_index, frame_index], dim=-1)
                    ind_list.append(ind.cpu().numpy())
                torch.cuda.empty_cache()
            
            result = {
                'episode': episode,
                'gt_episode': gt_episode,
                'obs_episode': obs_episode,
                'img_episode': img_episode,
                'ind_list': ind_list,
            }
            save_numpy(f'{prefix}_condition_sample', **result)

    # get_pred_gt_data()
    data=load_numpy(f'{prefix}_condition_sample')
    episode = np.stack(data['episode'], axis=0) # (num_segment, num_flows, num_step, horizon * 14)
    gt_episode = np.stack(data['gt_episode'], axis=0)[:,None,None,:]
    obs_episode = np.stack(data['obs_episode'], axis=0)[:,None,None,:]
    img_episode = np.stack(data['img_episode'], axis=0)[:,None,None,:]
    ind_list = np.concatenate(data['ind_list'], axis=0)

    num_segment, num_flows, num_step, feat_dim = episode.shape
    # action_pca, (gt_pca,) = pca(episode, gt_episode)
    action_pca, (gt_pca,) = tnse(episode, gt_episode)
    action_pca, gt_pca = action_pca[..., -1:,-1], gt_pca[...,-1] # (num_segment, num_flows, num_step)
    obs_pca = tnse(obs_episode)[0][...,-1] # (num_segment, num_flows, num_step)
    img_pca = tnse(img_episode)[0][...,-1] # (num_segment, num_flows, num_step)

    # mean = gt_pca.mean()
    # std = gt_pca.std()
    # norm_action = (action_pca - mean) / (std+1e-8)
    # norm_gt = (gt_pca - mean) / (std+1e-8)
    # norm_obs = standardize(obs_pca)
    # norm_img = standardize(img_pca)

    norm_action, norm_gt = min_max_normalize(action_pca, gt_pca)
    norm_obs, _ = min_max_normalize(obs_pca)
    norm_img, _ = min_max_normalize(img_pca)
    import matplotlib.pyplot as plt
    out_dir = './outputs/flows'
    os.makedirs(out_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(50, 50))
    # axes = axes[0]

    frame_mask = np.random.binomial(1, min(1700, num_segment)/num_segment, size=(num_segment,)).astype(bool)
    for l, cond in enumerate([norm_img, norm_obs]):
        l2_inverse = np.linalg.norm(x=(cond-norm_gt)[:,0,0], ord=2)
        l2_reverse = np.linalg.norm(x=(-cond-norm_gt)[:,0,0], ord=2)
        if l2_reverse < l2_inverse: cond = -cond
        cond_gt_mapping = np.concatenate([cond, norm_gt], axis=-1)
        cond_pred_mapping = np.concatenate([cond.repeat(num_flows, axis=1), norm_action], axis=-1)
        ax_up = axes[0][l]
        ax_down = axes[1][l]
        for i in range(num_segment):
            if not frame_mask[i]: continue
            time_steps = np.arange(cond_gt_mapping.shape[-1])
            ep, fr = ind_list[i]
            fr_max = ind_list[ind_list[:,0] == ep][-1,1]
            color = cm.get_cmap('cool')(fr*1.0/fr_max)
            for j in range(cond_gt_mapping.shape[1]):
                ax_up.plot(
                    time_steps, cond_gt_mapping[i, j, :], 
                    color=color, 
                    label=f'flow_{i}', linewidth=1, zorder=100, 
                    alpha=(1.)
                )
            for j in range(num_flows):
                ax_up.plot(
                    time_steps, cond_pred_mapping[i, j, :], 
                    color=color, 
                    label=f'flow_{i}', linewidth=5, zorder=1, 
                    alpha=(1./frame_mask.sum()/num_flows*20)
                )
                ax_down.plot(
                    time_steps, cond_pred_mapping[i, j, :], 
                    color=color, 
                    label=f'flow_{i}', linewidth=5, zorder=1, 
                    alpha=(1./frame_mask.sum()/num_flows*50)
                )

        ax_up.set_xlabel('Mapping')
        ax_down.set_xlabel('Mapping')
        ax_up.set_ylabel('PCA Flow Value')
        ax_down.set_ylabel('PCA Flow Value')
        ax_up.set_title(f'Normalized Flows' if l == 0 else 'Global Flows')
        ax_down.set_title(f'Normalized Flows' if l == 0 else 'Global Flows')
        ax_up.grid(True, linestyle='--', alpha=0.3)
        ax_down.grid(True, linestyle='--', alpha=0.3)
        ax_up.set_ylim(-2.5, 2.5)
        ax_down.set_ylim(-2.5, 2.5)
        # if num_flows <= 20:
        #     ax.legend(loc='upper right', fontsize='x-small', ncol=1)
    
    plt.tight_layout()
    out_path = os.path.join(out_dir, f'{prefix}_condition_sample_visualization.png')
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print("Flow visualizations saved.")

def visualize_obs(val_dataset, policy, prefix):
    horizon = policy.horizon
    n_obs_steps = policy.n_obs_steps
    n_act_steps = horizon - n_obs_steps
    policy.eval()

    def get_pred_gt_data():
        val_dataset.detail_item = True
        seq_sampler = SequentialStrideSampler(val_dataset, stride=7)
        batch_sampler = BatchSampler(seq_sampler, batch_size=1, drop_last=False)
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset,
            batch_sampler=batch_sampler,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
        )
        
        with torch.no_grad():
            with torch.autocast(device_type=str(policy.device), dtype=torch.bfloat16):
                episode = []
                gt_episode = []
                obs_episode = []
                img_episode = []
                ind_list = []
                for step, batch in enumerate(tqdm(val_dataloader)):
                    frame_index = batch['obs'].pop('frame_index')
                    episode_index = batch['obs'].pop('episode_index')
                    task_index = batch['obs'].pop('task_index')
                    sample_batch = {k: v.repeat(4, *([1] * (v.dim() - 1))) for k, v in batch['obs'].items()}
                    flows = []
                    for _ in range(1):
                        result = policy.predict_action(sample_batch, return_flow=True)
                        flow = result['flow']
                        flows.append(np.stack(flow, axis=1))
                    flows = np.concatenate(flows, axis=0)
                    flows = flows.reshape(*flows.shape[:-2], -1)  # (num_flows, num_step, horizon * 14)
                    episode.append(flows)
                    obs = result['obs'].to(device='cpu', dtype=torch.float32).numpy()[0].reshape(-1) # (1, 1, 4 * 1024)
                    obs_episode.append(obs)
                    img = result['img'].to(device='cpu', dtype=torch.float32).numpy()[0].reshape(-1)
                    img_episode.append(img)
                    gt = result['gt'].cpu().numpy()[0].reshape(-1)
                    gt_episode.append(gt)
                    ind = torch.stack([episode_index, frame_index], dim=-1)
                    ind_list.append(ind.cpu().numpy())
                torch.cuda.empty_cache()
            
            result = {
                'episode': episode,
                'gt_episode': gt_episode,
                'obs_episode': obs_episode,
                'img_episode': img_episode,
                'ind_list': ind_list,
            }
            save_numpy(f'{prefix}_condition_sample', **result)

    # get_pred_gt_data()
    data=load_numpy(f'{prefix}_condition_sample')
    episode = np.stack(data['episode'], axis=0) # (num_segment, num_flows, num_step, horizon * 14)
    gt_episode = np.stack(data['gt_episode'], axis=0)[:,None,None,:]
    obs_episode = np.stack(data['obs_episode'], axis=0)[:,None,None,:]
    img_episode = np.stack(data['img_episode'], axis=0)[:,None,None,:]
    ind_list = np.concatenate(data['ind_list'], axis=0)

    num_segment, num_flows, num_step, feat_dim = episode.shape
    # action_pca, (gt_pca,) = pca(episode, gt_episode)
    action_pca, (gt_pca,) = tnse(episode, gt_episode, n_components=1)
    # action_pca, gt_pca = action_pca[..., -1:,:], gt_pca # (num_segment, num_flows, num_step)
    obs_pca = tnse(obs_episode, n_components=2)[0] # (num_segment, num_flows, num_step, 2)
    img_pca = tnse(img_episode, n_components=2)[0] # (num_segment, num_flows, num_step, 2)

    norm_action, norm_gt = min_max_normalize(action_pca, gt_pca, 0., 1.)
    # norm_obs, _ = min_max_normalize(obs_pca)
    # norm_img, _ = min_max_normalize(img_pca)
    import matplotlib.pyplot as plt
    out_dir = './outputs/flows'
    os.makedirs(out_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(50, 50))
    # axes = axes[0]

    frame_mask = np.random.binomial(1, min(1700, num_segment)/num_segment, size=(num_segment,)).astype(bool)
    for l, cond in enumerate([img_pca, obs_pca]):
        ax_up = axes[0][l]
        ax_down = axes[1][l]
        # 绘制cond[:,-1,-1]的散点图
        for i in range(num_segment):
            # if not frame_mask[i]: continue
            ep, fr = ind_list[i]
            fr_max = ind_list[ind_list[:,0] == ep][-1,1]
            ind = fr * 1.0 / fr_max
            color = cm.get_cmap('cool')(ind)
            ax_up.scatter(
                cond[i, 0, 0, 0], cond[i, 0, 0, 1],
                color=color,
                label=f'point_{i}', s=50, zorder=(100*ind),
                alpha=(1.)
            )
            color = cm.get_cmap('rainbow')(norm_gt[i, 0, 0, 0])
            ax_down.scatter(
                cond[i, 0, 0, 0], cond[i, 0, 0, 1],
                color=color,
                label=f'point_{i}', s=50, zorder=(100*ind),
                alpha=(1.)
            )

    plt.tight_layout()
    out_path = os.path.join(out_dir, f'{prefix}_obs_visualization.png')
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print("Flow visualizations saved.")

def eval(args, logger):
    # task_name, expert_data_num, ckpt_setting, checkpoint_num
    set_seed(args.seed)
    prefix = f"{args.ckpt_setting}"
    checkpoint_dir = f"checkpoints/vodpp_{args.task_name}_{args.expert_data_num}_{args.ckpt_setting}"
    if not os.path.exists(checkpoint_dir):
        checkpoint_dir = f"checkpoints/vodpp/{args.task_name}_{args.expert_data_num}/{args.ckpt_setting}"
    if args.get('checkpoint_num', None) is not None:
        checkpoint_dir = f"{checkpoint_dir}/checkpoint-{args.checkpoint_num}"
        prefix = f"{prefix}_ckp{args.checkpoint_num}"
    policy_model = load_policy(checkpoint_dir, use_ckp_code=False, device='cuda')
    prefix = f"{prefix}_{args.task_name}"
    
    # Dataset and DataLoaders creation
    train_dataset = hydra.utils.instantiate(args.train_dataset)
    val_dataset = train_dataset.get_validation_dataset()
    # 实现batch_sampler要求顺序采样，batch size为1，stride为n_act_steps

    # visualize_flow(val_dataset, policy_model, prefix)
    # visualize_condition_flow(val_dataset, policy_model, prefix)
    # visualize_condition_sample_flow(val_dataset, policy_model, prefix)
    visualize_obs(val_dataset, policy_model, prefix)
    # visualize_traj(val_dataset, policy_model, prefix)

    # return dict(loss_for_log)
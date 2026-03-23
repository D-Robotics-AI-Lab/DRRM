import copy
import logging
import os
from pathlib import Path
import random
from xml.parsers.expat import model
import hydra
# import netron
import numpy as np
from omegaconf import OmegaConf

import diffusers
import torch
import torch.nn.functional as F

# --- Global Patch Start ---
# Force antialias=False to prevent "aten::_upsample_bilinear2d_aa" error during ONNX export
_original_interpolate = F.interpolate
def _global_interpolate_patched(input, size=None, scale_factor=None, mode='nearest', align_corners=None, recompute_scale_factor=None, antialias=False):
    return _original_interpolate(input, size=size, scale_factor=scale_factor, mode=mode, 
                               align_corners=align_corners, recompute_scale_factor=recompute_scale_factor, 
                               antialias=False)
F.interpolate = _global_interpolate_patched
print("Global patch: F.interpolate(antialias=True) forced to False for ONNX export.")
# --- Global Patch End ---

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
import onnxruntime as ort
from typing import Tuple

# # --- MonkeyPatch Start ---
# # Fix for ONNX export error: aten::_upsample_bilinear2d_aa not supported
# try:
#     import vggt.heads.dpt_head
#     import torch.nn.functional as F
    
#     def custom_interpolate_patched(
#         x: torch.Tensor,
#         size: Tuple[int, int] = None,
#         scale_factor: float = None,
#         mode: str = "bilinear",
#         align_corners: bool = True,
#     ) -> torch.Tensor:
#         """
#         Patched custom interpolate to force antialias=False for ONNX export.
#         """
#         if size is None and scale_factor is not None:
#             size = (int(x.shape[-2] * scale_factor), int(x.shape[-1] * scale_factor))
        
#         # Original logic handles INT_MAX, but for ONNX export we usually don't hit that constant folding size
#         # and we want clean graph.
#         return F.interpolate(x, size=size, mode=mode, align_corners=align_corners, antialias=False)

#     print("Monkey-patching vggt.heads.dpt_head.custom_interpolate for ONNX export...")
#     vggt.heads.dpt_head.custom_interpolate = custom_interpolate_patched
# except ImportError:
#     print("Could not import vggt.heads.dpt_head, skipping monkeypatch.")
# except Exception as e:
#     print(f"Failed to monkeypatch custom_interpolate: {e}")
# # --- MonkeyPatch End ---

# --- MonkeyPatch Start: RoPE ---
try:
    import vggt.layers.rope
    from typing import Tuple

    def patched_compute_frequency_components(
        self, dim: int, seq_len, device: torch.device, dtype: torch.dtype
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        is_symbolic = isinstance(seq_len, torch.Tensor)
        
        if not is_symbolic:
            cache_key = (dim, seq_len, device, dtype)
            if cache_key in self.frequency_cache:
                return self.frequency_cache[cache_key]
        
        exponents = torch.arange(0, dim, 2, device=device).float() / dim
        inv_freq = 1.0 / (self.base_frequency**exponents)
        
        # When seq_len is tensor, arange(seq_len) works
        positions = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
        angles = torch.einsum("i,j->ij", positions, inv_freq)
        
        angles = angles.to(dtype)
        angles = torch.cat((angles, angles), dim=-1)
        cos_components = angles.cos().to(dtype)
        sin_components = angles.sin().to(dtype)
        
        if not is_symbolic:
            self.frequency_cache[cache_key] = (cos_components, sin_components)
        
        return cos_components, sin_components

    def patched_forward_rope(self, tokens: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        feature_dim = tokens.size(-1) // 2
        
        # Use tensor operation for max_position to support dynamic shapes in export
        max_position = positions.max() + 1
        
        cos_comp, sin_comp = self._compute_frequency_components(feature_dim, max_position, tokens.device, tokens.dtype)
        
        vertical_features, horizontal_features = tokens.chunk(2, dim=-1)
        
        vertical_features = self._apply_1d_rope(vertical_features, positions[..., 0], cos_comp, sin_comp)
        horizontal_features = self._apply_1d_rope(horizontal_features, positions[..., 1], cos_comp, sin_comp)
        
        return torch.cat((vertical_features, horizontal_features), dim=-1)

    print("Monkey-patching vggt.layers.rope.RotaryPositionEmbedding2D for ONNX export...")
    vggt.layers.rope.RotaryPositionEmbedding2D._compute_frequency_components = patched_compute_frequency_components
    vggt.layers.rope.RotaryPositionEmbedding2D.forward = patched_forward_rope

except ImportError:
    print("Could not import vggt.layers.rope, skipping RoPE monkeypatch.")
except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"Failed to monkeypatch RoPE: {e}")
# --- MonkeyPatch End: RoPE ---

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

def export_basepolicy_to_onnx(model, dummy_input, onnx_save_path, opset_version=16):
    """
    将BasePolicy模型导出为ONNX（opset<=19）
    
    Args:
        model: 加载好权重的BasePolicy实例（eval模式）
        dummy_input: 匹配forward输入的示例张量（tuple/dict）
        onnx_save_path: ONNX文件保存路径（如'base_policy_opset19.onnx'）
        opset_version: ONNX算子集版本（<=19）
    """
    # 关键：设置为eval模式，关闭训练相关层（dropout/batchnorm）
    model.obs_encoder.scene_encoder.vggt_heads = None
    model.eval()
    # dummy_input = tuple(dummy_input.values())
    with torch.no_grad():  # 禁用梯度，加速导出并减少冗余
        # try:
        torch.onnx.export(
            model,
            kwargs=dummy_input,
            f=onnx_save_path,
            # 核心配置：指定opset版本<=19
            opset_version=opset_version,
            # 输入输出命名（方便后续部署识别）
            input_names=["head_cam", "agent_pos", "noise_trajectory"],
            output_names=["action", "action_pred"],
            # 动态维度配置（支持任意batch_size/seq_len，必配！）
            dynamic_axes={
                "head_cam": {0: "batch_size"},
                "agent_pos": {0: "batch_size"},
                "noise_trajectory": {0: "batch_size"},
                "action": {0: "batch_size"},
                "action_pred": {0: "batch_size"}
            },
            # 兼容性配置
            do_constant_folding=True,  # 常量折叠（提升推理效率）
            export_params=True,        # 导出权重参数（必须为True）
            verbose=False,             # 调试时可设为True查看导出日志
            keep_initializers_as_inputs=False,  # 优化权重存储
            dynamo=True,                 # 使用torchdynamo加速导出
            report=True
        )
        # except Exception as e:
        #     print(f"⚠️ ONNX导出失败：{e}")
    print(f"✅ 模型导出完成！路径：{onnx_save_path} | opset版本：{opset_version}")

def onnx(args, logger):
    # task_name, expert_data_num, ckpt_setting, checkpoint_num
    set_seed(args.seed)
    prefix = f"{args.ckpt_setting}"
    checkpoint_dir = f"checkpoints/vodp_{args.task_name}_{args.expert_data_num}_{args.ckpt_setting}"
    if not os.path.exists(checkpoint_dir):
        checkpoint_dir = f"checkpoints/vodp/{args.task_name}_{args.expert_data_num}/{args.ckpt_setting}"
    if args.get('checkpoint_num', None) is not None:
        checkpoint_dir = f"{checkpoint_dir}/checkpoint-{args.checkpoint_num}"
        prefix = f"{prefix}_ckp{args.checkpoint_num}"
    policy_model = load_policy(checkpoint_dir, use_ckp_code=False, device='cuda')
    policy_model = policy_model.float()
    prefix = f"{prefix}_opset19"
    
    # Dataset and DataLoaders creation
    train_dataset = hydra.utils.instantiate(args.train_dataset)
    val_dataset = train_dataset.get_validation_dataset()
    
    # val_dataset.detail_item = True
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
        with torch.autocast(device_type=str(policy_model.device), dtype=torch.float32):
            samples = []
            for step, batch in enumerate(tqdm(val_dataloader)):
                if step >= 100: break  # 仅验证一个batch
                # sample_batch = {k: v.repeat(4, *([1] * (v.dim() - 1))).float() for k, v in batch['obs'].items()}
                sample_batch = {k: v.float()[:,:1,...] for k, v in batch['obs'].items()}
                gt = batch['obs']['agent_pos'].cpu().numpy()
                sample_batch.pop('endpose')
                sample_batch.pop('front_cam')
                sample_batch['noise_trajectory'] = torch.randn(
                    1, policy_model.n_action_steps, policy_model.action_dim, 
                    device=policy_model.device
                )

                # result = policy_model.predict_action(sample_batch)
                result = policy_model(**sample_batch)
                result = [v.cpu().numpy() for k, v in result.items()]

                # export_basepolicy_to_onnx(
                #     model=policy_model,
                #     dummy_input=sample_batch,
                #     onnx_save_path=f"./onnx/{prefix}.onnx",
                #     opset_version=19
                # )

                # ONNX Runtime推理结果
                ort_sess = ort.InferenceSession(
                    f"./onnx/{prefix}.onnx",
                    providers=["CPUExecutionProvider"]  # GPU可用："CUDAExecutionProvider"
                )
                ort_inputs = {k: v.cpu().numpy() for k, v in sample_batch.items()}
                ort_out = ort_sess.run(["action", "action_pred"], ort_inputs)
                samples.append({**ort_inputs})
                
                # 数值一致性校验（误差<1e-5即合格）
                # np.testing.assert_allclose(gt, ort_out[0], rtol=1e-5, atol=1e-5)
                # np.testing.assert_allclose(gt, ort_out[1], rtol=1e-5, atol=1e-5)
                # print("✅ ONNX模型验证通过！PyTorch与ONNX推理结果一致")
            if not os.path.exists(f"./onnx/{prefix}_samples"):
                os.mkdir(f"./onnx/{prefix}_samples")
            for i, item in enumerate(samples):
                # os.mkdir(f"./onnx/{prefix}_samples/{i}")
                for k in item.keys():
                    np.save(f"./onnx/{prefix}_samples/{i}/{k}.npy", item[k])
            print("ONNX export and inference test completed.")


# def onnx(args, logger):
#     sess = ort.InferenceSession("film_23d_1f_opset19.onnx", providers=['CPUExecutionProvider'])
#     input_names = [inp.name for inp in sess.get_inputs()]
#     output_names = [out.name for out in sess.get_outputs()]
#     # 构造输入
#     data1 = np.random.rand(1, 1, 3, 240, 320).astype(np.float32)
#     data2 = np.random.rand(1, 1, 3, 240, 320).astype(np.float32)
#     data3 = np.random.rand(1, 1, 14).astype(np.float32)
#     # 组织输入字典
#     input_feed = {
#         input_names[0]: data1,
#         input_names[1]: data2,
#         input_names[2]: data3,
#     }
#     # 推理
#     output = sess.run(output_names, input_feed)
#     print("infer success")

#     # model_path = "film_23d_1f_opset19.onnx"
#     # # 检查文件是否存在，避免报错
#     # if os.path.exists(model_path):
#     #     print(f"正在加载模型: {model_path}")
#     #     # 启动 Netron 可视化
#     #     netron.start(model_path, address=("localhost", 8081), browse=False)
#     # else:
#     #     print(f"错误：找不到文件 '{model_path}'。请确保该文件与脚本在同一目录下，或提供绝对路径。")
#     # input()
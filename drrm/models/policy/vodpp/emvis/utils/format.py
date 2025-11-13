from numbers import Number
import os
import sys
import torch

def format_number(num: Number) -> str:
    if num >= 1e9:
        return f"{num / 1e9:.2f}B"  # 十亿
    elif num >= 1e6:
        return f"{num / 1e6:.2f}M"  # 百万
    elif num >= 1e3:
        return f"{num / 1e3:.2f}K"  # 千
    else:
        return f"{num:,}"
    
def print_weight(v, k=''):
    print(
        f"[{k}] \nMin: {v.min():.3f}, "
        f"Max: {v.max():.3f}, "
        f"Mean: {v.mean() if v.mean()!= 0. else 0:.3f}, "
        f"Std: {v.std():.3f}, "
        f"Type: {v.dtype}, "
        f"Requires Grad: {v.requires_grad}"
    )

def print_param(model, path = None, detail = False):
    if path and os.path.exists(os.path.split(path)[0]):
        sys.stdout = open(path, 'w')
    v_min, v_max = (torch.inf, None), (-torch.inf, None)
    o_min, o_max = (torch.inf, None), (-torch.inf, None)
    for k,v in model.named_parameters():
        if detail: print_weight(v, k)
        if v.min() < v_min[0]: v_min = (v.min(), k)
        if v.max() > v_max[0]: v_max = (v.max(), k)
        if v.min() < o_min[0] and not 'vggt_encoder' in k: o_min = (v.min(), k)
        if v.max() > o_max[0] and not 'vggt_encoder' in k: o_max = (v.max(), k)
    t_min = v_min if v_min[0] < o_min[0] else o_min
    t_max = v_max if v_max[0] > o_max[0] else o_max
    summary = (
        f"VGGT Min: {v_min[0]:.3f}, [{v_min[1]}] \n"
        f"VGGT Max: {v_max[0]:.3f}, [{v_max[1]}] \n"
        f"Others Min: {o_min[0]:.3f}, [{o_min[1]}] \n"
        f"Others Max: {o_max[0]:.3f}, [{o_max[1]}] \n"
        f"Total Min: {t_min[0]:.3f}, [{t_min[1]}] \n"
        f"Total Max: {t_max[0]:.3f}, [{t_max[1]}] \n"
    )
    sys.stdout = sys.__stdout__
    if path and os.path.exists(os.path.split(path)[0]):
        with open(path, 'r+') as f:
            text = f.read()
            f.seek(0)
            f.write(summary+'\n')
            f.write(text)
    else:
        print(summary)

def param_count(model, detail = False) -> str:
    v_min, v_max = (torch.inf, None), (-torch.inf, None)
    o_min, o_max = (torch.inf, None), (-torch.inf, None)
    detail = ''
    try:
        for k,v in model.named_parameters():
            if detail:
                result += (
                    f"\n[{k}] \nMin: {v.min():.3f}, "
                    f"Max: {v.max():.3f}, "
                    f"Mean: {v.mean() if v.mean()!= 0. else 0:.3f}, "
                    f"Std: {v.std():.3f}, "
                    f"Type: {v.dtype}, "
                    f"Requires Grad: {v.requires_grad}"
                )
            if 'vggt_encoder' in k:
                if v.min() < v_min[0]: v_min = (v.min(), k)
                if v.max() > v_max[0]: v_max = (v.max(), k)
            else:
                if v.min() < o_min[0]: o_min = (v.min(), k)
                if v.max() > o_max[0]: o_max = (v.max(), k)
        t_min = v_min if v_min[0] < o_min[0] else o_min
        t_max = v_max if v_max[0] > o_max[0] else o_max
        summary = (
            f"VGGT Min: {v_min[0]:.3f}, [{v_min[1]}] \n"
            f"VGGT Max: {v_max[0]:.3f}, [{v_max[1]}] \n"
            f"Others Min: {o_min[0]:.3f}, [{o_min[1]}] \n"
            f"Others Max: {o_max[0]:.3f}, [{o_max[1]}] \n"
            f"Total Min: {t_min[0]:.3f}, [{t_min[1]}] \n"
            f"Total Max: {t_max[0]:.3f}, [{t_max[1]}] \n"
        )
    except Exception as e:
        summary = ''
    return summary+detail
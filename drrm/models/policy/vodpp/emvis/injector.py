from typing import List
import torch
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict
from .layers import MLP, CrossAttention, get_proj_layer, DimFuser, AttnFuser

class DimInjector(nn.Module):
    def __init__(
        self,
        fuse_2d: bool,
        fuse_3d: bool = True,
        query2d: bool = False,
        # attn_3d: bool = False,
        dim_3d: int = 2048,
        dim_2d: int = 1024,
        dim_out: int = None,
        mlp_ratio: float = 1.0,
        drop_p: float = 0.,
        ffn_layer_num: int = 1,
        intermediate_layer_idx: List[int] = [4, 11, 17, 23],
        ** kwargs
    ):
        super().__init__()
        assert len(intermediate_layer_idx) >= 1
        if dim_out == None: dim_out = dim_3d
        self.intermediate_layer_idx = intermediate_layer_idx
        self.fuse_3d = fuse_3d
        self.fuse_2d = fuse_2d
        self.query2d = query2d
        # self.attn_3d = attn_3d

        # VGGT Fusion
        if fuse_3d:
            fuser_config = {
                'input_dim_list': [(idx, dim_3d) for idx in self.intermediate_layer_idx],
                'dim_out': dim_out,
                'mlp_ratio': mlp_ratio,
                'ffn_layer_num': ffn_layer_num,
                'drop_p': drop_p,
            }
            self.vggt_fuser = DimFuser(**fuser_config)
        else: 
            def vggt_fuser(x):
                assert len(self.intermediate_layer_idx) == 1
                x = x.squeeze(0)
                if dim_3d != dim_out:
                    pool = nn.AdaptiveAvgPool1d(dim_out//2)
                    x = torch.cat([
                        pool(x[...,:dim_3d//2]), 
                        pool(x[...,dim_3d//2:])
                    ], dim=-1)
                return x
            self.vggt_fuser = vggt_fuser
        # # VGGT Attn
        # fuser_config = {
        #     'input_dim_list': [('3D', dim_out), ('3D', dim_out)],
        #     'dim_out': dim_out,
        #     'mlp_ratio': mlp_ratio,
        #     'ffn_layer_num': ffn_layer_num,
        #     'drop_p': drop_p,
        #     ** kwargs
        # }
        # self.vggt_attn = AttnFuser(**fuser_config) if attn_3d else nn.Identity
        # 2D 3D Fusion
        fuser_config = {
            'input_dim_list': [('3D', dim_out), ('2D', dim_2d)],
            'dim_out': dim_out,
            'mlp_ratio': mlp_ratio,
            'ffn_layer_num': ffn_layer_num,
            'drop_p': drop_p,
            ** kwargs
        }
        # self.modality_fuser = DimFuser(**fuser_config) if fuse_2d else nn.Identity
        self.modality_fuser = AttnFuser(**fuser_config) if fuse_2d else nn.Identity

        self.post_norm = nn.LayerNorm(dim_out) if ffn_layer_num != 0 and (fuse_3d or fuse_2d) else nn.Identity()
    
    def forward(self, tokens_3d_list, tokens_2d = None, pos_3d = None, pos_2d = None):
        """
        Args:
            tokens_2d: [B * S, V2d, P, D2d]
            tokens_3d_list: [L, B * S, V3d, P, D3d]
        Returns: 
            fused_tokens: [B * S, V, P, D]
        """
        # Format 3D tokens
        L, BS, V3d, P, D3d = tokens_3d_list.shape
        P3d = V3d * P
        input_tokens_list = tokens_3d_list.view((L, BS, V3d * P, D3d)) # [L, B * S, V3d * P, D3d]
        if pos_3d != None:
            pos_3d = pos_3d.contiguous().view(BS, -1, 2) # [B * S, V3d * P, 2]
            _, p_num, _ = pos_3d.shape
            assert p_num == P3d
        # Format 2D tokens
        if tokens_2d != None:
            if self.query2d:
                tokens_2d = tokens_2d[:,0:1,...]
                pos_2d = pos_2d[:,0:1,...]
            _, V2d, _, D2d = tokens_2d.shape
            P2d = V2d * P
            tokens_2d = tokens_2d.contiguous().view(BS, -1, D2d) # [B * S, V2d * P, D2d]
            if pos_2d != None:
                pos_2d = pos_2d.contiguous().view(BS, -1, 2) # [B * S, V2d * P, 2]
                _, p_num, _ = pos_2d.shape
                assert p_num == P2d


        # VGGT Fusion
        fused_tokens = self.vggt_fuser(input_tokens_list) # [B * S, V3d * P, D]
        # # VGGT Attn
        # if self.attn_3d:
        #     fused_tokens = self.vggt_attn(fused_tokens, fused_tokens, pos_3d, pos_3d)
        # 2D 3D Fusion
        if self.fuse_2d:
            if isinstance(self.modality_fuser, DimFuser):
                fused_tokens = self.modality_fuser([fused_tokens, tokens_2d])
            elif isinstance(self.modality_fuser, AttnFuser):
                if self.query2d:
                    fused_tokens = self.modality_fuser(tokens_2d, fused_tokens, pos_2d, pos_3d) # 2D query 3D
                else:
                    fused_tokens = self.modality_fuser(fused_tokens, tokens_2d, pos_3d, pos_2d) # 3D query 2D
        
        fused_tokens = self.post_norm(fused_tokens)
        _, _, D = fused_tokens.shape
        return fused_tokens.view(BS, V3d, P, D) # [B * S, V3d, P, D]
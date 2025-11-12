from typing import List
import torch
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict
from .mlp import get_proj_layer

class DimFuser(nn.Module):
    def __init__(
        self,
        input_dim_list: List[tuple[str, int]], # [('key1', dim1), ('key2', dim2)]
        dim_out: int,
        mlp_ratio: float = 1.0,
        drop_p: float = 0.,
        ffn_layer_num: int = 1,
        ** kwargs
    ):
        super().__init__()
        self.dim_concat = sum(list(zip(*input_dim_list))[1])
        self.dim_out = dim_out
        self.layers_list = [*input_dim_list]
        self.ffn_on = ffn_layer_num != 0

        # Pre-normalization for VGGT layers
        self.pre_norm_layers = nn.ModuleList([
            nn.Sequential(
                OrderedDict([(f'nrom_{key}', nn.LayerNorm(dim))])
            )
            for key, dim in self.layers_list
        ])
        # Dimension projection for concatenated VGGT layers
        # dim_concat = self.dim_concat
        proj_config = {
            '__target__': 'Nonlinear',
            'dim_in': self.dim_concat,
            'dim_out': self.dim_out
        }
        self.dim_fusion = get_proj_layer(**proj_config)
        # Dimension projection for concatenated VGGT layers
        proj_config = {
            '__target__': 'MLP',
            'dim_in': self.dim_out,
            'dim_out': self.dim_out,
            'mlp_ratio': mlp_ratio,
            'layer_num': ffn_layer_num,
            'drop_p': drop_p
        }
        self.ffn_fusion = get_proj_layer(**proj_config)
        # FFN Residual
        self.dropout = nn.Dropout(drop_p)

    def forward(self, input_tokens):
        """
        Args:
            input_tokens: List[Tensor] L * [B * S, V * P, D_l]
        Returns: 
            fused_tokens: [B * S, V, P, D_o]
        """
        BS, VP, _ = input_tokens[0].shape
        
        tokens_list = []
        # Pre-normalization
        for idx, pre_norm in enumerate(self.pre_norm_layers):
            tokens_list.append(pre_norm(input_tokens[idx])) # [B * S, V * P, D_l]

        # Dimension projection
        concat_tokens = torch.concat(tokens_list, dim=-1)
        fused_tokens = self.dim_fusion(concat_tokens) # [B * S, V * P, L * D]

        # FFN Residual
        if self.ffn_on:
            ffn_tokens = self.ffn_fusion(fused_tokens) # [B * S, V * P, L * D]
            fused_tokens = fused_tokens + self.dropout(ffn_tokens)
        
        return fused_tokens

class AttnFuser(nn.Module):
    def __init__(
        self, 
        input_dim_list: List[tuple[str, int]], # [('key1', dim1), ('key2', dim2)]
        dim_out: int,
        num_heads=8, 
        mlp_ratio=4.0,
        rope=None,
        norm_layers: List = ['attn', 'ffn'],
        residual: bool = True,
        ffn_layer_num: int = 1,
        drop_p: float = 0.,
        ** kwargs
    ):
        super().__init__()
        dim_q, dim_c = list(zip(*input_dim_list))[1]
        dim = dim_out
        self.Wq = nn.Linear(dim_q, dim)
        self.Wk = nn.Linear(dim_c, dim)
        self.Wv = nn.Linear(dim_c, dim)
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q_norm = nn.LayerNorm(self.head_dim) if 'attn' in norm_layers else nn.Identity()
        self.k_norm = nn.LayerNorm(self.head_dim) if 'attn' in norm_layers else nn.Identity()
        self.rope = rope
        ffn_config = {
            '__target__': 'MLP',
            'dim_in': dim,
            'dim_out': dim,
            'mlp_ratio': mlp_ratio,
            'layer_num': ffn_layer_num,
            'drop_p': drop_p
        }
        self.ffn = get_proj_layer(**ffn_config)
        self.ffn_norm = nn.LayerNorm(dim) if 'ffn' in norm_layers else nn.Identity()
        self.residual = residual

    def forward(self, query, context, qpos, cpos):
        # B, V, P, D = query.shape # [B, Vq, Pq, D]
        B, N, D = query.shape # [B, Vq*Pq, D]
        _, M, _ = context.shape # [B, Vc*Pc, D]
        d = self.head_dim
        # Multi head cross attention
        res = query
        Q = self.Wq(query)  # [B, Px, D]
        K = self.Wk(context)  # [B, Pc, D]
        V = self.Wv(context)  # [B, Pc, D]
        Q = Q.view(B, N, self.num_heads, self.head_dim).transpose(1,2)  # [B, H, Px, d]
        K = K.view(B, M, self.num_heads, self.head_dim).transpose(1,2)
        V = V.view(B, M, self.num_heads, self.head_dim).transpose(1,2)
        Q, K = self.q_norm(Q), self.k_norm(K) # norm
        if self.rope is not None:
            Q = self.rope(Q, qpos)
            K = self.rope(K, cpos)
        # Scaled Dot-Product Attention
        attn = (Q @ K.transpose(-2, -1)) / (d ** 0.5)  # [B, H, Px, Pc]
        attn = attn.softmax(dim=-1)
        attn_out = (attn @ V).transpose(1,2).reshape(B, N, D) # [B, Px, D]
        if self.residual: attn_out = res + attn_out
        # FFN
        res = attn_out
        attn_out = self.ffn_norm(attn_out)
        attn_out = self.ffn(attn_out)
        if self.residual: attn_out = res + attn_out
        
        return attn_out

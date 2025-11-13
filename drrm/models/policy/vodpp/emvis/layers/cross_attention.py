from typing import List
from torch import nn
from .mlp import MLP

class CrossAttention(nn.Module):
    def __init__(
        self, 
        dim, 
        num_heads=8, 
        mlp_ratio=4.0,
        rope=None,
        norm_layers: List = ['attn', 'ffn'],
        residual: bool = True,
        ** kwargs
    ):
        super().__init__()
        self.Wq = nn.Linear(dim, dim)
        self.Wk = nn.Linear(dim, dim)
        self.Wv = nn.Linear(dim, dim)
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q_norm = nn.LayerNorm(self.head_dim) if 'attn' in norm_layers else nn.Identity()
        self.k_norm = nn.LayerNorm(self.head_dim) if 'attn' in norm_layers else nn.Identity()
        self.rope = rope
        self.ffn = MLP(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            out_features=dim
        )
        self.ffn_norm = nn.LayerNorm(dim) if 'ffn' in norm_layers else nn.Identity()
        self.residual = residual

    def forward(self, query, context, qpos, cpos):
        B, N, D = query.shape # [B, Px, D]
        d = self.head_dim
        _, M, _ = context.shape # [B, Pc, D]
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
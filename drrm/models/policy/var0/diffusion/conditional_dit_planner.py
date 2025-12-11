# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DiT: https://github.com/facebookresearch/DiT
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------
from collections import OrderedDict

import torch
import torch.nn as nn

from .blocks import (Block, TimestepEmbedder,
                               get_1d_sincos_pos_embed_from_grid,
                               get_multimodal_cond_pos_embed)


class DiTPlanner(nn.Module):
    """
    Class for Robotics Diffusion Transformers.
    """
    def __init__(
        self,
        # output_dim: int,
        # horizon: int,
        hidden_size=1024,
        depth=8,
        num_heads=16,
        self_attn_first=False,
        gen_len=4096, # S * V * P
        gen_pe_config=None, # S, V, P
        cond_len=4096, # S * V * P
        cond_pe_config=None, # S, V, P
        dtype=torch.bfloat16
    ):
        super().__init__()
        # self.horizon = horizon
        self.hidden_size = hidden_size
        self.gen_len = gen_len
        self.gen_pe_config = gen_pe_config
        self.cond_len = cond_len
        self.cond_pe_config = cond_pe_config
        self.dtype = dtype

        self.t_embedder = TimestepEmbedder(hidden_size, dtype=dtype)
        
        # We will use trainable sin-cos embeddings
        # [timestep; state; action]
        self.gen_pos_embed = nn.Parameter(
            torch.zeros(1, gen_len+1, hidden_size))
        # Image conditions
        self.cond_pos_embed = nn.Parameter(
            torch.zeros(1, cond_len, hidden_size))
        
        self.blocks = nn.ModuleList([
            Block(hidden_size, num_heads, self_attn_first) for _ in range(depth)
        ])
        # self.final_layer = FinalLayer(hidden_size, output_dim)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize pos_embed by sin-cos embedding
        gen_pos_embed = get_multimodal_cond_pos_embed(
            embed_dim=self.hidden_size,
            mm_cond_lens=OrderedDict([
                ('timestep', 1),
                *self.gen_pe_config,
            ])
        )
        self.gen_pos_embed.data.copy_(torch.from_numpy(gen_pos_embed).float().unsqueeze(0))

        cond_pos_embed = get_multimodal_cond_pos_embed(
            embed_dim=self.hidden_size,
            mm_cond_lens=OrderedDict(self.cond_pe_config),
            embed_modality=False
        )
        self.cond_pos_embed.data.copy_(torch.from_numpy(cond_pos_embed).float().unsqueeze(0))

        # Initialize timestep embedding MLP
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        
        # Initialize the final layer: zero-out the final linear layer
        # nn.init.constant_(self.final_layer.ffn_final.fc2.weight, 0)
        # nn.init.constant_(self.final_layer.ffn_final.fc2.bias, 0)
        
        # Move all the params to given data type:
        self.to(self.dtype)

    def forward(self, x, t, cond, cond_mask=None):
        """
        Forward pass of RDT.
        
        x: (B, T, D), state + action token sequence, T = horizon + 1,
            dimension D is assumed to be the same as the hidden size.
        freq: (B,), a scalar indicating control frequency.
        t: (B,) or (1,), diffusion timesteps.
        lang_c: (B, L_lang, D) or None, language condition tokens (variable length),
            dimension D is assumed to be the same as the hidden size.
        img_c: (B, L_img, D) or None, image condition tokens (fixed length),
            dimension D is assumed to be the same as the hidden size.
        lang_mask: (B, L_lang) or None, language condition mask (True for valid).
        img_mask: (B, L_img) or None, image condition mask (True for valid).
        """
        t = self.t_embedder(t).unsqueeze(1)             # (B, 1, D) or (1, 1, D)
        # Append timestep to the input tokens
        if t.shape[0] == 1:
            t = t.expand(x.shape[0], -1, -1)
        x = torch.cat([t, x], dim=1)               # (B, T, D)
        
        # Add multimodal position embeddings
        x = x + self.gen_pos_embed
        # Note the lang is of variable length
        cond = cond + self.cond_pos_embed

        # Forward pass
        for i, block in enumerate(self.blocks):
            c, mask = cond, cond_mask
            x = block(x, c, mask)                       # (B, T+1, D)

        return x[:, 1:, :]
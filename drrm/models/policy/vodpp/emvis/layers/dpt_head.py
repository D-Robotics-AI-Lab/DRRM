# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


# Inspired by https://github.com/DepthAnything/Depth-Anything-V2


import os
from typing import List, Dict, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from vggt.heads.dpt_head import DPTHead as VGGTDPTHead
from vggt.heads.dpt_head import custom_interpolate, activate_head

class DPTHead(VGGTDPTHead):
    def forward(
        self, *args, **kwargs
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if 'images' in kwargs:
            return super(DPTHead, self).forward(*args, **kwargs)
        else:
            return self.shape_forward(*args, **kwargs)

    def shape_forward(
        self,
        aggregated_tokens_list: List[torch.Tensor],
        # images: torch.Tensor,
        # shape: Tuple[int, int],
        pos: torch.Tensor,
        patch_start_idx: int,
        frames_chunk_size: int = 8,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through the DPT head, supports processing by chunking frames.
        Args:
            aggregated_tokens_list (List[Tensor]): List of token tensors from different transformer layers.
            images (Tensor): Input images with shape [B, S, 3, H, W], in range [0, 1].
            patch_start_idx (int): Starting index for patch tokens in the token sequence.
                Used to separate patch tokens from other tokens (e.g., camera or register tokens).
            frames_chunk_size (int, optional): Number of frames to process in each chunk.
                If None or larger than S, all frames are processed at once. Default: 8.

        Returns:
            Tensor or Tuple[Tensor, Tensor]:
                - If feature_only=True: Feature maps with shape [B, S, C, H, W]
                - Otherwise: Tuple of (predictions, confidence) both with shape [B, S, 1, H, W]
        """
        shape = (pos[0,0,:,1]==0).sum().item() * self.patch_size, (pos[0,0,:,0]==0).sum().item() * self.patch_size
        # B, S, _, H, W = images.shape
        S = aggregated_tokens_list[self.intermediate_layer_idx[0]].shape[1]

        # If frames_chunk_size is not specified or greater than S, process all frames at once
        if frames_chunk_size is None or frames_chunk_size >= S:
            return self._forward_impl(aggregated_tokens_list, shape, patch_start_idx)

        # Otherwise, process frames in chunks to manage memory usage
        assert frames_chunk_size > 0

        # Process frames in batches
        all_features = []
        all_preds = []
        all_conf = []

        for frames_start_idx in range(0, S, frames_chunk_size):
            frames_end_idx = min(frames_start_idx + frames_chunk_size, S)

            # Process batch of frames
            if self.feature_only:
                chunk_output = self._forward_impl(
                    aggregated_tokens_list, shape, patch_start_idx, frames_start_idx, frames_end_idx
                )
                all_preds.append(chunk_output)
            else:
                chunk_feature, chunk_preds, chunk_conf = self._forward_impl(
                    aggregated_tokens_list, shape, patch_start_idx, frames_start_idx, frames_end_idx
                )
                all_features.append(chunk_feature)
                all_preds.append(chunk_preds)
                all_conf.append(chunk_conf)

        # Concatenate results along the sequence dimension
        if self.feature_only:
            return torch.cat(all_preds, dim=1)
        else:
            return torch.cat(all_features, dim=1), torch.cat(all_preds, dim=1), torch.cat(all_conf, dim=1)
        
    def _forward_impl(
        self, *args, **kwargs
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if 'images' in kwargs or not isinstance(args[1], Tuple):
            return super(DPTHead, self)._forward_impl(*args, **kwargs)
        else:
            return self._shape_forward_impl(*args, **kwargs)

    def _shape_forward_impl(
        self,
        aggregated_tokens_list: List[torch.Tensor],
        # images: torch.Tensor,
        shape: Tuple[int, int],
        patch_start_idx: int,
        frames_start_idx: int = None,
        frames_end_idx: int = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Implementation of the forward pass through the DPT head.

        This method processes a specific chunk of frames from the sequence.

        Args:
            aggregated_tokens_list (List[Tensor]): List of token tensors from different transformer layers.
            images (Tensor): Input images with shape [B, S, 3, H, W].
            patch_start_idx (int): Starting index for patch tokens.
            frames_start_idx (int, optional): Starting index for frames to process.
            frames_end_idx (int, optional): Ending index for frames to process.

        Returns:
            Tensor or Tuple[Tensor, Tensor]: Feature maps or (predictions, confidence).
        """
        B, S = aggregated_tokens_list[self.intermediate_layer_idx[0]].shape[:2]
        if frames_start_idx is not None and frames_end_idx is not None:
        #     images = images[:, frames_start_idx:frames_end_idx].contiguous()
            S = frames_end_idx - frames_start_idx

        # B, S, _, H, W = images.shape
        H, W = shape
        

        patch_h, patch_w = H // self.patch_size, W // self.patch_size

        out = []
        dpt_idx = 0

        for layer_idx in self.intermediate_layer_idx:
            x = aggregated_tokens_list[layer_idx][:, :, patch_start_idx:]

            # Select frames if processing a chunk
            if frames_start_idx is not None and frames_end_idx is not None:
                x = x[:, frames_start_idx:frames_end_idx]

            x = x.view(B * S, -1, x.shape[-1])

            x = self.norm(x)

            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))

            x = self.projects[dpt_idx](x)
            if self.pos_embed:
                x = self._apply_pos_embed(x, W, H)
            x = self.resize_layers[dpt_idx](x)

            out.append(x)
            dpt_idx += 1

        # Fuse features from multiple layers.
        out = self.scratch_forward(out)
        feature = out.view((B, S, *out.shape[1:]))
        if self.feature_only:
            return feature
        
        # Interpolate fused output to match target image resolution.
        out = custom_interpolate(
            out,
            (int(patch_h * self.patch_size / self.down_ratio), int(patch_w * self.patch_size / self.down_ratio)),
            mode="bilinear",
            align_corners=True,
        )

        if self.pos_embed:
            out = self._apply_pos_embed(out, W, H)

        # if self.feature_only:
        #     return out.view(B, S, *out.shape[1:])

        out = self.scratch.output_conv2(out)
        preds, conf = activate_head(out, activation=self.activation, conf_activation=self.conf_activation)

        preds = preds.view(B, S, *preds.shape[1:])
        conf = conf.view(B, S, *conf.shape[1:])
        return feature, preds, conf
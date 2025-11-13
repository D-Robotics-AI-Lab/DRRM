import logging
import os
import re
import time
from typing import Dict, List, Tuple, Union
import torch
from vggt.models.aggregator import Aggregator, slice_expand_and_flatten

logger = logging.getLogger(__name__)
_URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
AA_pattern = r'(?:global_blocks|frame_blocks)\.(\d+)\.'

class VGGTEncoder(Aggregator):
    def __init__(self, *args, **kwargs):
        self.ft_layer_idx = kwargs.pop('ft_layer_idx', [])
        self.dim_keys = kwargs.pop('dim_keys', [])
        self.intermediate_layer_idx = kwargs.pop('intermediate_layer_idx')
        super().__init__(*args, **kwargs)
        self.ft_param_keys = []
        for key, _ in self.named_parameters():
            match = re.search(AA_pattern, key)
            if match and int(match.group(1)) in self.ft_layer_idx:
                self.ft_param_keys.append(key)

    def load_pretrained_model(
            self,
            model_path: str = None
    ):
        logger.info(f"Loading VGGT Encoder......")
        if model_path and os.path.exists(model_path):
            state_dict = torch.load(model_path)
        else:
            state_dict = torch.hub.load_state_dict_from_url(_URL)
        state_dict = self.state_dict_filter(state_dict)
        missing_keys, unexpected_keys = self.load_state_dict(state_dict)
        logger.info(f"Missing keys: {missing_keys}")
        logger.info(f"Unexpected keys: {unexpected_keys}")
        # frozen parameters
        for key, param in self.named_parameters():
            if not key in self.ft_param_keys:
                param.requires_grad = False
            else:
                logger.debug(f"Trainable Parameter: {key}")

    def state_dict_filter(
        self,
        state_dict: Dict[str, torch.Tensor]
    ):
        state_dict = {
            k[k.find('.')+1:]: v
            for k, v in state_dict.items() if 'aggregator' in k
        }
        return state_dict
        
    def aggregator_forward(
        self,
        images: torch.Tensor,
    ) -> Tuple[List[torch.Tensor], int]:
        """
        Args:
            images (torch.Tensor): Input images with shape [B, S, 3, H, W], in range [0, 1].
                B: batch size, S: sequence length, 3: RGB channels, H: height, W: width

        Returns:
            (list[torch.Tensor], int):
                The list of outputs from the attention blocks,
                and the patch_start_idx indicating where patch tokens begin.
        """
        B, S, C_in, H, W = images.shape

        if C_in != 3:
            raise ValueError(f"Expected 3 input channels, got {C_in}")

        # Normalize images and reshape for patch embed
        images = (images - self._resnet_mean) / self._resnet_std

        # Reshape to [B*S, C, H, W] for patch embedding
        images = images.view(B * S, C_in, H, W)
        patch_tokens = self.patch_embed(images)

        if isinstance(patch_tokens, dict):
            patch_tokens = patch_tokens["x_norm_patchtokens"]

        _, P, C = patch_tokens.shape

        # Expand camera and register tokens to match batch size and sequence length
        camera_token = slice_expand_and_flatten(self.camera_token, B, S)
        register_token = slice_expand_and_flatten(self.register_token, B, S)

        # Concatenate special tokens with patch tokens
        tokens = torch.cat([camera_token, register_token, patch_tokens], dim=1)

        pos, patch_pos = None, None
        if self.rope is not None:
            patch_pos = self.position_getter(B * S, H // self.patch_size, W // self.patch_size, device=images.device)

            if self.patch_start_idx > 0:
                # do not use position embedding for special tokens (camera and register tokens)
                # so set pos to 0 for the special tokens
                pos = patch_pos + 1
                pos_special = torch.zeros(B * S, self.patch_start_idx, 2).to(images.device).to(pos.dtype)
                pos = torch.cat([pos_special, pos], dim=1)

        # update P because we added special tokens
        _, P, C = tokens.shape

        frame_idx = 0
        global_idx = 0
        output_list = []

        for _ in range(self.aa_block_num):
            for attn_type in self.aa_order:
                if attn_type == "frame":
                    tokens, frame_idx, frame_intermediates = self._process_frame_attention(
                        tokens, B, S, P, C, frame_idx, pos=pos
                    )
                elif attn_type == "global":
                    tokens, global_idx, global_intermediates = self._process_global_attention(
                        tokens, B, S, P, C, global_idx, pos=pos
                    )
                else:
                    raise ValueError(f"Unknown attention type: {attn_type}")

            for i in range(len(frame_intermediates)):
                # concat frame and global intermediates, [B x S x P x 2C]
                concat_inter = torch.cat([frame_intermediates[i], global_intermediates[i]], dim=-1)
                output_list.append(concat_inter)

        del concat_inter
        del frame_intermediates
        del global_intermediates
        return output_list, patch_tokens.view(B, S, -1, C), patch_pos.view(B, S, -1, 2)

    def forward(
        self,
        images: torch.Tensor,
        V_fake: int = None,
    ) -> Dict[str, Union[torch.Tensor, List[torch.Tensor]]]:
        """
        Forward pass of the VGGT model.

        Args:
            images (torch.Tensor): Input images with shape [S, 3, H, W] or [B, S, 3, H, W], in range [0, 1].
                B: batch size, S: sequence length, 3: RGB channels, H: height, W: width

        Returns:
            dict: A dictionary containing the following predictions:
                - Image_tokens (torch.Tensor): Image Encoding tokens with shape [B, S, P, D]
                - camera_tokens_list (List[torch.Tensor]): Camera Encoding tokens with shape [24, B, S, D]
                - scene_tokens_list (List[torch.Tensor]): Scene Encoding tokens with shape [24, B, S, P, D]
                
        """
        if len(images.shape) == 4:
            images = images.unsqueeze(0)
        BS, V, C_in, H, W = images.shape
        if V_fake: images = images.view((-1, V_fake, C_in, H, W))

        (
            aggregated_tokens_list, 
            patch_tokens, 
            patch_pos
        ) = self.aggregator_forward(images)

        patch_tokens = patch_tokens.reshape((BS, V, *patch_tokens.shape[2:]))
        patch_pos = patch_pos.reshape((BS, V, *patch_pos.shape[2:]))
        aggregated_tokens_list = [
            v.reshape((BS, V, *v.shape[2:])) 
            for v in aggregated_tokens_list
        ]
        dim_pos = []
        for key in self.dim_keys:
            if key == 'frame': 
                dim_pos += list(range(0,1024))
            elif key == 'global':
                dim_pos += list(range(1024,2048))
            else:
                raise KeyError()

        output = {
            'image_tokens':patch_tokens,
            'image_tokens_pos':patch_pos,
            'camera_tokens_list': {
                idx: aggregated_tokens_list[idx][...,0:1,:]
                for idx in self.intermediate_layer_idx
            }, 
            'spatial_tokens_pos': patch_pos,
            'spatial_tokens_list': {
                idx: aggregated_tokens_list[idx][...,self.patch_start_idx:,dim_pos]
                for idx in self.intermediate_layer_idx
            }
        }
        
        return output
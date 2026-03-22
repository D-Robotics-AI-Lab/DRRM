import logging
import os
import re
import time
from typing import Dict, List, Tuple, Union
import numpy as np
import torch 
from torch import nn
from transformers import pipeline
from transformers import AutoImageProcessor, AutoModel, AutoConfig
from transformers import DINOv3ViTModel
from transformers.image_utils import load_image

logger = logging.getLogger(__name__)
model_name = "facebook/dinov3-vith16plus-pretrain-lvd1689m"
AA_pattern = r'(?:global_blocks|frame_blocks)\.(\d+)\.'

class PositionGetter:
    """Generates and caches 2D spatial positions for patches in a grid.

    This class efficiently manages the generation of spatial coordinates for patches
    in a 2D grid, caching results to avoid redundant computations.

    Attributes:
        position_cache: Dictionary storing precomputed position tensors for different
            grid dimensions.
    """

    def __init__(self):
        """Initializes the position generator with an empty cache."""
        self.position_cache: Dict[Tuple[int, int], torch.Tensor] = {}

    def __call__(self, batch_size: int, height: int, width: int, device: torch.device) -> torch.Tensor:
        """Generates spatial positions for a batch of patches.

        Args:
            batch_size: Number of samples in the batch.
            height: Height of the grid in patches.
            width: Width of the grid in patches.
            device: Target device for the position tensor.

        Returns:
            Tensor of shape (batch_size, height*width, 2) containing y,x coordinates
            for each position in the grid, repeated for each batch item.
        """
        if (height, width) not in self.position_cache:
            y_coords = torch.arange(height, device=device)
            x_coords = torch.arange(width, device=device)
            positions = torch.cartesian_prod(y_coords, x_coords)
            self.position_cache[height, width] = positions

        cached_positions = self.position_cache[height, width]
        return cached_positions.view(1, height * width, 2).expand(batch_size, -1, -1).clone()

class DINOv3Encoder(DINOv3ViTModel):
    def __init__(self, config: AutoConfig, *args, **kwargs):
        super().__init__(config)
        self.position_getter = PositionGetter()
        self.custom_patch_size = 16
        # self.ft_layer_idx = kwargs.pop('ft_layer_idx', [])
        # self.dim_keys = kwargs.pop('dim_keys', [])
        # self.intermediate_layer_idx = kwargs.pop('intermediate_layer_idx', [23])
        
        # self.model = AutoModel.from_pretrained(
        #     model_name,
        #     # device_map="auto",
        # )
        # super().__init__(*args, **kwargs)

    def load_pretrained_model(
            self,
            model_name: str = None
    ):
        # from safetensors.torch import load_file
        # logger.info(f"Loading DA3 Encoder......")
        pretrained = AutoModel.from_pretrained(model_name)
        self.load_state_dict(pretrained.state_dict())
        # del pretrained
        for key, param in self.named_parameters():
            param.requires_grad = False
        

    def forward(
        self,
        images: torch.Tensor,
        V_fake: int = None,
    ) -> Dict[str, Union[torch.Tensor, List[torch.Tensor]]]:
        """
        Forward pass of the DA3 model.

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
        # if V_fake: images = images.view((-1, V_fake, C_in, H, W))
        
        images = images.reshape(-1, C_in, H, W)
        feats = super().forward(images).last_hidden_state
        patch_pos = self.position_getter(BS*V, H // self.custom_patch_size, W // self.custom_patch_size, device=images.device)
        sp_len = self.embeddings.cls_token.shape[1] + self.embeddings.register_tokens.shape[1]
        patch_pos = patch_pos.reshape((BS, V, *patch_pos.shape[-2:]))

        output = {
            'image_tokens': feats[:, sp_len:, :],
            'image_tokens_pos': patch_pos,
            'camera_tokens_list': None, 
            'spatial_tokens_pos': patch_pos,
            'spatial_tokens_list': None
        }
        
        return output
    
    def da3_preprocess(self, images: torch.Tensor, process_res: int = 504) -> torch.Tensor:
        """Preprocess input images for DA3 model.

        Args:
            images (torch.Tensor): Input images with shape [B, S, 3, H, W], in range [0, 1].
                B: batch size, S: sequence length, 3: RGB channels, H: height, W: width
        Returns:
            torch.Tensor: Preprocessed images with shape [B*S, 3, H, W], in range [0, 1].
        """
        images = images.cpu().numpy()
        images = [(images[i]*255).astype(np.uint8).transpose(1, 2, 0) for i in range(images.shape[0])]
        imgs_cpu, extrinsics, intrinsics = self._preprocess_inputs(
            images, None, None, process_res, "upper_bound_resize"
        )
        images, ex_t, in_t = self._prepare_model_inputs(imgs_cpu, extrinsics, intrinsics)
        images = images.reshape((-1, *images.shape[2:]))
        return images
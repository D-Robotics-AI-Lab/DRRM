import logging
import os
import re
import time
from typing import Dict, List, Tuple, Union
import numpy as np
import torch
# from depth_anything_3.cfg import create_object, load_config
from depth_anything_3.api import DepthAnything3
# from depth_anything_3.registry import MODEL_REGISTRY
# from depth_anything_3.utils.io.input_processor import InputProcessor
# from depth_anything_3.utils.io.output_processor import OutputProcessor

logger = logging.getLogger(__name__)
model_name = "da3nested-giant-large"
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

class DA3Encoder(DepthAnything3):
    def __init__(self, *args, **kwargs):
        self.ft_layer_idx = kwargs.pop('ft_layer_idx', [])
        self.dim_keys = kwargs.pop('dim_keys', [])
        self.intermediate_layer_idx = kwargs.pop('intermediate_layer_idx', [23])
        if "model_name" not in kwargs:
            kwargs["model_name"] = model_name
        self.position_getter = PositionGetter()
        self.patch_size = 14
        super().__init__(*args, **kwargs)

    def load_pretrained_model(
            self,
            model_name: str = None
    ):
        # from safetensors.torch import load_file
        logger.info(f"Loading DA3 Encoder......")
        pretrained = self.from_pretrained(model_name)
        self.load_state_dict(pretrained.state_dict())
        del pretrained
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
        if V_fake: images = images.view((-1, V_fake, C_in, H, W))
        
        feats, aux_feats = self.model.da3.backbone(
            images, cam_token=None, export_feat_layers=[]
        )
        aggregated_tokens_list = [feat[0] for feat in feats]
        patch_pos = self.position_getter(BS*V, H // self.patch_size, W // self.patch_size, device=images.device)

        patch_pos = patch_pos.reshape((BS, V, *patch_pos.shape[-2:]))
        aggregated_tokens_list = [
            v.reshape((BS, V, *v.shape[2:])) 
            for v in aggregated_tokens_list
        ]
        dim_pos = []
        for key in self.dim_keys:
            if key == 'frame': 
                dim_pos += list(range(0,1536))
            elif key == 'global':
                dim_pos += list(range(1536,3072))
            else:
                raise KeyError()

        output = {
            'image_tokens': aggregated_tokens_list[-1][...,dim_pos],
            'image_tokens_pos': patch_pos,
            'camera_tokens_list': feats[-1][1], 
            'spatial_tokens_pos': patch_pos,
            'spatial_tokens_list': {
                idx: aggregated_tokens_list[idx][...,dim_pos]
                for idx in self.intermediate_layer_idx
            }
        }
        
        return output
    
    def da3_preprocess(self, images: torch.Tensor) -> torch.Tensor:
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
            images, None, None, 504, "upper_bound_resize"
        )
        images, ex_t, in_t = self._prepare_model_inputs(imgs_cpu, extrinsics, intrinsics)
        images = images.reshape((-1, *images.shape[2:]))
        return images
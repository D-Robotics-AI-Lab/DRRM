import logging
import os
import time
from typing import Dict, List
import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin  # used for model hub

from vggt.heads.camera_head import CameraHead
# from vggt.heads.dpt_head import DPTHead as MetaDPTHead
from vggt.heads.track_head import TrackHead
from .layers import DPTHead

logger = logging.getLogger(__name__)
_URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
HEADS = ['camera_head', 'point_head', 'depth_head', 'track_head']

class VGGTHead(nn.Module, PyTorchModelHubMixin):
    def __init__(
            self, 
            patch_size=14, 
            embed_dim=1024, 
            heads=HEADS, 
            ft_heads: List = []
        ):
        super().__init__()
        self.ft_heads = ft_heads
        if 'camera_head' in heads:
            self.camera_head = CameraHead(dim_in=2 * embed_dim)
        if 'point_head' in heads:
            self.point_head = DPTHead(dim_in=2 * embed_dim, output_dim=4, activation="inv_log", conf_activation="expp1")
        if 'depth_head' in heads:
            self.depth_head = DPTHead(dim_in=2 * embed_dim, output_dim=2, activation="exp", conf_activation="expp1")
        if 'track_head' in heads:
            self.track_head = TrackHead(dim_in=2 * embed_dim, patch_size=patch_size)

    def load_pretrained_model(
            self,
            model_path: str = None
    ):
        logger.info(f"Loading VGGT Heads......")
        if model_path and os.path.exists(model_path):
            state_dict = torch.load(model_path)
        else:
            state_dict = torch.hub.load_state_dict_from_url(_URL)
        state_dict = self.state_dict_filter(state_dict)
        missing_keys, unexpected_keys = self.load_state_dict(state_dict)
        logger.info(f"Missing keys: {missing_keys}")
        logger.info(f"Unexpected keys: {unexpected_keys}")

        def include_ft_key(k):
            for exp in self.ft_heads:
                if exp in k and not 'scratch.output_conv2' in k: return True
            return False
        for key, param in self.named_parameters():
            if not include_ft_key(key):
                param.requires_grad = False

    def state_dict_filter(
        self,
        state_dict: Dict[str, torch.Tensor]
    ):
        exp_keys = [head for head in HEADS if not hasattr(self,head)]
        def has_exp_key(k):
            for exp in exp_keys:
                if exp in k: return True
            return False
        filted_state_dict = {}
        for k, v in state_dict.items():
            if 'aggregator' in k: continue
            elif has_exp_key(k): continue
            filted_state_dict[k] = v

        return filted_state_dict
    
    def forward(
        self,
        images: torch.Tensor,
        camera_tokens_list: Dict[str, torch.Tensor],
        spatial_tokens_list: Dict[str, torch.Tensor],
        query_points: torch.Tensor = None,
        ** kwargs
    ):
        """
        Forward pass of the VGGT model.

        Args:
            images (torch.Tensor): Input images with shape [S, 3, H, W] or [B, S, 3, H, W], in range [0, 1].
                B: batch size, S: sequence length, 3: RGB channels, H: height, W: width
            query_points (torch.Tensor, optional): Query points for tracking, in pixel coordinates.
                Shape: [N, 2] or [B, N, 2], where N is the number of query points.
                Default: None

        Returns:
            dict: A dictionary containing the following predictions:
                - pose_enc (torch.Tensor): Camera pose encoding with shape [B, S, 9] (from the last iteration)
                - depth (torch.Tensor): Predicted depth maps with shape [B, S, H, W, 1]
                - depth_conf (torch.Tensor): Confidence scores for depth predictions with shape [B, S, H, W]
                - world_points (torch.Tensor): 3D world coordinates for each pixel with shape [B, S, H, W, 3]
                - world_points_conf (torch.Tensor): Confidence scores for world points with shape [B, S, H, W]
                - images (torch.Tensor): Original input images, preserved for visualization

                If query_points is provided, also includes:
                - track (torch.Tensor): Point tracks with shape [B, S, N, 2] (from the last iteration), in pixel coordinates
                - vis (torch.Tensor): Visibility scores for tracked points with shape [B, S, N]
                - conf (torch.Tensor): Confidence scores for tracked points with shape [B, S, N]
        """

        # If without batch dimension, add it
        if len(images.shape) == 4:
            images = images.unsqueeze(0)
        if query_points is not None and len(query_points.shape) == 2:
            query_points = query_points.unsqueeze(0)
        
        patch_start_idx = 0
        predictions = {}
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        with torch.autocast(str(images.device), dtype=dtype):
            if self.camera_head is not None and camera_tokens_list:
                last_cam_layer = list(camera_tokens_list.values())[-1]
                if len(last_cam_layer.shape) == 3:
                    last_cam_layer = last_cam_layer[:,:,None,:]
                last_cam_layer = [last_cam_layer]
                pose_enc_list = self.camera_head(last_cam_layer)
                predictions["pose_enc"] = pose_enc_list[-1]  # pose encoding of the last iteration

            if self.depth_head is not None:
                depth, depth_conf = self.depth_head(
                    spatial_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["depth"] = depth
                predictions["depth_conf"] = depth_conf

            if self.point_head is not None:
                pts3d, pts3d_conf = self.point_head(
                    spatial_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["world_points"] = pts3d
                predictions["world_points_conf"] = pts3d_conf

        if self.track_head is not None and query_points is not None:
            track_list, vis, conf = self.track_head(
                spatial_tokens_list, images=images, patch_start_idx=patch_start_idx, query_points=query_points
            )
            predictions["track"] = track_list[-1]  # track of the last iteration
            predictions["vis"] = vis
            predictions["conf"] = conf

        predictions["images"] = images

        return predictions
    


if __name__ == "__main__":
    heads = VGGTHead()
    heads.load_pretrained_model(os.environ['VGGT_CKP'])

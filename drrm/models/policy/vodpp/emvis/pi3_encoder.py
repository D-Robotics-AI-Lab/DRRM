import logging
import os
import re
import time
from typing import Dict, List, Tuple, Union
import torch
from pi3 import Pi3
from pi3.utils.geometry import homogenize_points

logger = logging.getLogger(__name__)
_URL = "https://huggingface.co/yyfz233/Pi3/blob/main/model.safetensors"

class Pi3Encoder(Pi3):
    def __init__(self, *args, **kwargs):
        self.ft_layer_idx = kwargs.pop('ft_layer_idx', [])
        self.dim_keys = kwargs.pop('dim_keys', [])
        self.intermediate_layer_idx = kwargs.pop('intermediate_layer_idx', [23])
        super().__init__(*args, **kwargs)

    def load_pretrained_model(
            self,
            model_path: str = None
    ):
        from safetensors.torch import load_file
        logger.info(f"Loading Pi3 Encoder......")
        if model_path and os.path.exists(model_path):
            # state_dict = load_file(model_path)
            pretrained = self.from_pretrained(model_path)
        else:
            pretrained = self.from_pretrained("yyfz233/Pi3")
        self.load_state_dict(pretrained.state_dict())
        del pretrained
        for key, param in self.named_parameters():
            param.requires_grad = False

    def pi3_encode(self, imgs):
        imgs = (imgs - self.image_mean) / self.image_std

        B, N, _, H, W = imgs.shape
        
        # encode by dinov2
        imgs = imgs.reshape(B*N, _, H, W)
        hidden = self.encoder(imgs, is_training=True)

        if isinstance(hidden, dict):
            hidden = hidden["x_norm_patchtokens"]
        sem_hidden = hidden

        hidden, pos = self.decode(hidden, N, H, W)

        return hidden, sem_hidden, pos
    
    def get_pi3_heads(self, **kwargs):
        return Pi3Heads(
            self.patch_start_idx,
            self.point_decoder,
            self.point_head,
            self.conf_decoder,
            self.conf_head,
            self.camera_decoder,
            self.camera_head,
            **kwargs
        )
        
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
            hidden, sem_tokens, pos
        ) = self.pi3_encode(images)
        
        dim_pos = []
        for key in self.dim_keys:
            if key == 'frame': 
                dim_pos += list(range(0,1024))
            elif key == 'global':
                dim_pos += list(range(1024,2048))
            else:
                raise KeyError()
        
        sem_tokens = sem_tokens.reshape((BS, V, *sem_tokens.shape[1:]))
        geo_tokens = hidden[:,self.patch_start_idx:,dim_pos]
        geo_tokens = geo_tokens.reshape((BS, V, *geo_tokens.shape[1:]))
        patch_pos = pos[:,self.patch_start_idx:,...] - 1
        patch_pos = patch_pos.reshape((BS, V, *patch_pos.shape[1:]))

        output = {
            'image_tokens':sem_tokens,
            'image_tokens_pos':patch_pos,
            'camera_tokens_list': None, 
            'spatial_tokens_pos': patch_pos,
            'spatial_tokens_list': {
                23:  geo_tokens # 35
            },
            'hidden': hidden,
            'pos': pos,
        }
        
        return output
    
class Pi3Heads(torch.nn.Module):
    def __init__(
        self, 
        patch_start_idx,
        point_decoder,
        point_head,
        conf_decoder,
        conf_head,
        camera_decoder,
        camera_head,
        **kwargs
    ):
        super().__init__()
        self.point_decoder = point_decoder
        self.point_head = point_head
        self.conf_decoder = conf_decoder
        self.conf_head = conf_head
        self.camera_decoder = camera_decoder
        self.camera_head = camera_head
        self.patch_start_idx = patch_start_idx

    def load_pretrained_model(self, *args, **kwargs):
        pass

    def forward(
        self,
        images: torch.Tensor,
        hidden,
        pos,
        ** kwargs
    ):
        B, N, _, H, W = images.shape
        patch_h, patch_w = H // 14, W // 14
        predictions = {}

        point_hidden = self.point_decoder(hidden, xpos=pos)
        conf_hidden = self.conf_decoder(hidden, xpos=pos)
        camera_hidden = self.camera_decoder(hidden, xpos=pos)
        with torch.amp.autocast(device_type='cuda', enabled=False):
            # local points
            point_hidden = point_hidden.float()
            ret = self.point_head([point_hidden[:, self.patch_start_idx:]], (H, W)).reshape(B, N, H, W, -1)
            xy, z = ret.split([2, 1], dim=-1)
            z = torch.exp(z)
            local_points = torch.cat([xy * z, z], dim=-1)
            predictions["depth"] = z

            # confidence
            conf_hidden = conf_hidden.float()
            conf = self.conf_head([conf_hidden[:, self.patch_start_idx:]], (H, W)).reshape(B, N, H, W, -1)
            predictions["depth_conf"] = conf
            predictions["world_points_conf"] = conf

            # camera
            camera_hidden = camera_hidden.float()
            camera_poses = self.camera_head(camera_hidden[:, self.patch_start_idx:], patch_h, patch_w).reshape(B, N, 4, 4)
            predictions["pose_enc"] = camera_poses

            # unproject local points using camera poses
            points = torch.einsum('bnij, bnhwj -> bnhwi', camera_poses, homogenize_points(local_points))[..., :3]
            predictions["world_points"] = points

        predictions["images"] = images

        return predictions

    
if __name__ == "__main__":
    pi3 = Pi3Encoder()
    pi3.load_pretrained_model()
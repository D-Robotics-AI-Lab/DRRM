from typing import Dict, Tuple, Union
import copy
from sympy.core.symbol import Str
import torch
import torch.nn as nn
import torchvision

from drrm.models.policy.diffusion_policy.vision.crop_randomizer import CropRandomizer
from drrm.models.policy.diffusion_policy.common.pytorch_util import dict_apply, replace_submodules
from drrm.models.policy.diffusion_policy.common.module_attr_mixin import ModuleAttrMixin

from drrm.emvis import EmVisRM


class MultiImageObsEncoderEmvisExp1(ModuleAttrMixin):
    def __init__(self,
        emvis_config: dict = None,
        out_channels: int = None,
        rgb_key: str = 'head_cam',
        state_key: str = 'agent_pos',
        **kwargs
        ):
        super().__init__()
        self.state_key = state_key
        self.out_channels = out_channels
        self.rgb_key = rgb_key
        self.scene_encoder = EmVisRM(**emvis_config)

    def forward(self, obs_dict):
        batch_size = None
        features = list()

        # process vggt input or rgb input
        if "spatial_tokens_4" in obs_dict.keys() or "spatial_tokens_11" in obs_dict.keys() or "spatial_tokens_17" in obs_dict.keys() or "spatial_tokens_23" in obs_dict.keys() \
            or "camera_tokens_4" in obs_dict.keys() or "camera_tokens_11" in obs_dict.keys() or "camera_tokens_17" in obs_dict.keys() or "camera_tokens_23" in obs_dict.keys() \
            or "image_tokens" in obs_dict.keys() or "image_tokens_pos" in obs_dict.keys():

            spatial_tokens_list_keys = ['spatial_tokens_23', 'spatial_tokens_17', 'spatial_tokens_11', 'spatial_tokens_4']
            camera_tokens_list_keys = ['camera_tokens_23', 'camera_tokens_17', 'camera_tokens_11', 'camera_tokens_4']
            spatial_tokens_list = {}
            camera_tokens_list = {}
            image_tokens = None
            image_tokens_pos = None
            for key, value in obs_dict.items():
                if key in spatial_tokens_list_keys:
                    # 获取字符串中最后一个_对应的数字
                    idx = int(key.split('_')[-1])
                    spatial_tokens_list[idx] = value.unsqueeze(1)
                elif key in camera_tokens_list_keys:
                    idx = int(key.split('_')[-1])
                    camera_tokens_list[idx] = value.unsqueeze(1)
                elif key == 'image_tokens':
                    image_tokens = value.unsqueeze(1)
                elif key == 'image_tokens_pos':
                    image_tokens_pos = value.unsqueeze(1)
            vggt_tokens_dict = {
                'spatial_tokens_list': spatial_tokens_list,
                'camera_tokens_list': camera_tokens_list,
                'image_tokens': image_tokens,
                'image_tokens_pos': image_tokens_pos,
                'spatial_tokens_pos': image_tokens_pos,
            }
            emvis_feat = self.scene_encoder(vggt_token_dict=vggt_tokens_dict).squeeze(1).squeeze(1)
            features.append(emvis_feat)
        else:
            rgb_image = obs_dict[self.rgb_key].unsqueeze(1)
            # 重塑图像形状并归一化到0-1范围
            rgb_image = (rgb_image + 1) / 2  # 从[-1,1]归一化到[0,1]
            emvis_feat = self.scene_encoder(rgb_image).squeeze(1).squeeze(1)
            features.append(emvis_feat)
        
        # process lowdim input
        agent_pos = obs_dict[self.state_key]
        features.append(agent_pos)
        
        # concatenate all features
        result = torch.cat(features, dim=-1)
        return result
    
    def output_shape(self):
        return torch.Size([self.out_channels])    # TODO: 改成自动推导

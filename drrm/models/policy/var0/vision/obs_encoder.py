from typing import Dict, Tuple, Union
import copy
from sympy.core.symbol import Str
import torch
import torch.nn as nn
import torchvision

from .crop_randomizer import CropRandomizer
from ..common.pytorch_util import dict_apply, replace_submodules
from ..common.module_attr_mixin import ModuleAttrMixin

from ..emvis import EmVisRM


class SceneEncoder(ModuleAttrMixin):
    def __init__(
        self,
        emvis_config: dict = None,
        shape_meta: dict = None,
        rgb_key: list[str] = ['head_cam'],
        state_key: str = 'agent_pos',
        **kwargs
    ):
        super().__init__()
        self.state_key = state_key
        self.rgb_key = rgb_key
        self.scene_encoder = EmVisRM(**emvis_config)
        
        self.out_view_meta = 1 \
            if self.scene_encoder.mv_fuser != None \
                or self.scene_encoder.only_first_view \
            else len(rgb_key)
        self.out_shape_meta = self.scene_encoder.shape_out
        self.out_dim_meta = self.scene_encoder.dim_out \
            + (shape_meta['obs'][state_key]['shape'][0] if state_key else 0)

    def forward(self, obs_dict) -> torch.tensor:
        batch_size = None
        features = list()
        BS = next(iter(obs_dict.values())).shape[0]
        dim = self.scene_encoder.dim_out

        # process rgb input
        rgb_image = torch.cat([obs_dict[key].unsqueeze(1) for key in self.rgb_key], dim=1) # BS, V, C, H, W 
        # 重塑图像形状并归一化到0-1范围
        rgb_image = (rgb_image + 1) / 2  # 从[-1,1]归一化到[0,1]
        emvis_feat = self.scene_encoder(rgb_image) # BS, V, C, H, W -> BS, V, 1, dim
        out_format =  (BS, -1) if self.state_key else (BS, -1, dim) # BS, V*P, dim / BS, dim
        emvis_feat = emvis_feat.reshape(*out_format)
        features.append(emvis_feat)
        
        # process lowdim input
        if self.state_key:
            agent_pos = obs_dict[self.state_key]
            features.append(agent_pos)
        
        # concatenate all features
        result = torch.cat(features, dim=-1)  # 512 * 2 + 14 = 1038
        return result
    
    def output_shape_meta(self):
        """
        return:
            Views, H, W, Channels
        """
        return torch.Size([self.out_view_meta, *self.out_shape_meta, self.out_dim_meta]) 

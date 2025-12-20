import re
from typing import Dict
import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_dpmsolver_multistep import \
    DPMSolverMultistepScheduler

from drrm.models.base_policy import BasePolicy
from .diffusion.conditional_dit_head import DiT
from .vision.obs_encoder import SceneEncoder as VODPPlusEncoder
from .diffusion.conditional_unet1d import ConditionalUnet1D
from .diffusion.mask_generator import LowdimMaskGenerator
from .common.normalizer import LinearNormalizer
from .common.pytorch_util import dict_apply
from .common.module_attr_mixin import ModuleAttrMixin

import yaml
import json
from dataclasses import dataclass
from typing import Optional
from transformers import PretrainedConfig, PreTrainedModel

@dataclass
class VODPPlusDitDDPMConfig(PretrainedConfig):
    shape_meta: dict
    noise_scheduler: DDPMScheduler
    noise_scheduler_sample: DPMSolverMultistepScheduler
    obs_encoder: VODPPlusEncoder
    hidden_size: int
    depth: int
    num_heads: int
    horizon: int
    n_action_steps: int
    n_obs_steps: int
    obs_as_global_cond: bool = True
    diffusion_step_embed_dim: int = 256
    down_dims: tuple = (256,512,1024)
    kernel_size: int = 5
    n_groups: int = 8
    cond_predict_scale: bool = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.auto_map = {}
        self.pkg_map = {}
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    @classmethod
    def from_customed_yaml(cls, yaml_path: str):
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)
    
    @classmethod
    def from_customed_json(cls, json_path: str):
        with open(json_path, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)
    
    @classmethod
    def from_customed_dict(cls, config_dict):
        return cls(**config_dict)


class VODPPlusDitDDPM(BasePolicy, PreTrainedModel, ModuleAttrMixin):
    config_class = VODPPlusDitDDPMConfig

    def __init__(self, config: VODPPlusDitDDPMConfig):
        super().__init__(config)
        self.num_inference_timesteps = config.noise_scheduler['num_inference_timesteps']
        noise_scheduler = {
            k:v for k, v in
            config.noise_scheduler.items() 
            if k != 'num_inference_timesteps'
        }
        self.noise_scheduler = hydra.utils.instantiate(noise_scheduler)
        self.noise_scheduler_sample = hydra.utils.instantiate(config.noise_scheduler_sample)
        self.obs_encoder = hydra.utils.instantiate(config.obs_encoder)

        action_shape = config.shape_meta['action']['shape']
        horizon = config.horizon
        hidden_size = config.hidden_size
        n_action_steps = config.n_action_steps
        n_obs_steps = config.n_obs_steps
        # parse shapes
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        # get feature dim
        V, H, W, dim = self.obs_encoder.output_shape_meta() # V, H, W, D
        obs_feature_dim = dim

        # create diffusion model
        num_patches = H * W
        scene_cond_len = (config.n_obs_steps * V * num_patches)
        scene_pos_embed_config = [("image", (config.n_obs_steps, V, num_patches)),]
        self.model = DiT(
            output_dim=action_dim,
            horizon=horizon,
            hidden_size=hidden_size,
            depth=config.depth,
            num_heads=config.num_heads,
            scene_cond_len=scene_cond_len,
            scene_pos_embed_config=scene_pos_embed_config
        )

        self.scene_adaptor = self.build_condition_adapter(
            config.scene_adaptor, 
            in_features=obs_feature_dim, 
            out_features=hidden_size
        )

        # A `state` refers to an action or a proprioception vector
        self.state_adaptor = self.build_condition_adapter(
            config.state_adaptor, 
            in_features=action_dim,    # state + state mask (indicator)
            out_features=hidden_size
        )

        # self.mask_generator = LowdimMaskGenerator(
        #     action_dim=action_dim,
        #     obs_dim=0 if obs_as_global_cond else obs_feature_dim,
        #     max_n_obs_steps=n_obs_steps,
        #     fix_obs_steps=True,
        #     action_visible=False
        # )
        self.normalizer = LinearNormalizer()
        # self.normalizer = None
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.action_mask = torch.zeros(1, horizon, action_dim).bool()
        self.action_mask[:,n_obs_steps:,:] = True
        # self.kwargs = kwargs
        self.kwargs = {} #

    def build_condition_adapter(
        self, projector_type, in_features, out_features):
        projector = None
        if projector_type == 'linear':
            projector = nn.Linear(in_features, out_features)
        else:
            mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
            if mlp_gelu_match:
                mlp_depth = int(mlp_gelu_match.group(1))
                modules = [nn.Linear(in_features, out_features)]
                for _ in range(1, mlp_depth):
                    modules.append(nn.GELU(approximate="tanh"))
                    modules.append(nn.Linear(out_features, out_features))
                projector = nn.Sequential(*modules)

        if projector is None:
            raise ValueError(f'Unknown projector type: {projector_type}')

        return projector
    
    def adapt_conditions(self, scene_tokens):
        '''
        scene_cond: (batch_size, patch_num, img_token_dim)
        
        return: adpated (..., hidden_size) for all input tokens
        '''
        adpated_scene = self.scene_adaptor(scene_tokens)
        return adpated_scene
    
    # ========= inference  ============
    def conditional_sample(
            self, scene_cond, state_traj, action_mask, **kwargs
        ) -> torch.Tensor:
        '''
        scene_cond: image conditional data, (batch_size, patch_num, hidden_size).
        state_traj: (batch_size, 1, hidden_size), state trajectory.
        
        return: (batch_size, horizon, action_dim)
        '''
        device = state_traj.device
        dtype = state_traj.dtype
        batch_size = state_traj.shape[0]
        noisy_action = torch.randn(
            size=(batch_size, self.horizon, self.action_dim), 
            dtype=dtype, device=device
        )

        # Set step values
        self.noise_scheduler_sample.set_timesteps(self.num_inference_timesteps)

        state_mask = ~action_mask
        for t in self.noise_scheduler_sample.timesteps:
            # Prepare state-action trajectory
            noisy_action[state_mask] = state_traj[state_mask]
            state_action_traj = self.state_adaptor(noisy_action)
            
            # Predict the model output
            model_output = self.model(state_action_traj, 
                                      t.unsqueeze(-1).to(device), scene_cond)
            
            # Compute previous actions: x_t -> x_t-1
            noisy_action = self.noise_scheduler_sample.step(
                model_output, t, noisy_action).prev_sample
            noisy_action = noisy_action.to(state_traj.dtype)
        
        # Finally apply the action mask to mask invalid action dimensions
        noisy_action[state_mask] = state_traj[state_mask]

        return noisy_action


    def predict_action(self, obs_dict: Dict[str, torch.Tensor], return_sample:bool = False) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        assert 'past_action' not in obs_dict # not implemented yet
        # normalize input
        filtered_obs_dict = {key: value for key, value in obs_dict.items() 
                if key in self.normalizer.params_dict}
        nobs = self.normalizer.normalize(filtered_obs_dict)
        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        Do = self.obs_feature_dim
        To = self.n_obs_steps

        # build input
        device = self.device
        dtype = self.dtype

        # handle different ways of passing observation
        # condition through global feature
        this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
        nobs_features = self.obs_encoder(this_nobs) # (BS, VP, Do)
        scene_cond = self.adapt_conditions(nobs_features) # (BS, VP, hidden_size)
        state_traj = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
        state_traj[:,:To,:] = this_nobs['agent_pos']
        action_mask = self.action_mask.expand(B, -1, -1).to(device=device)

        # run sampling
        nsample = self.conditional_sample(scene_cond, state_traj, action_mask)
        
        # unnormalize prediction
        naction_pred = nsample[...,:Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        # get action
        start = To
        end = start + self.n_action_steps
        action = action_pred[:,start:end]
        
        result = {
            'action': action,
            'action_pred': action_pred
        }
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        # normalize input
        assert 'valid_mask' not in batch
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])
        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]

        # handle different ways of passing observation
        # reshape B, T, ... to B*T
        this_nobs = dict_apply(nobs, lambda x: x[:,:self.n_obs_steps,...].reshape(-1,*x.shape[2:]))
        nobs_features = self.obs_encoder(this_nobs) # (BS, VP, Do)
        scene_cond = self.adapt_conditions(nobs_features) # (BS, VP, hidden_size)
        state_traj = nactions
        action_mask = self.action_mask.expand(batch_size, -1, -1).to(device=state_traj.device)

        # Sample noise that we'll add to the images
        noise = torch.randn(state_traj.shape, device=state_traj.device)
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (batch_size,), device=nactions.device
        ).long()
        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_traj = self.noise_scheduler.add_noise(
            state_traj, noise, timesteps)
        
        # compute loss mask
        cond_mask = ~action_mask

        # apply conditioning
        noisy_traj[cond_mask] = state_traj[cond_mask]
        state_noisy_tokens = self.state_adaptor(noisy_traj)
        
        # Predict the noise residual
        pred = self.model(state_noisy_tokens, timesteps, scene_cond)

        pred_type = self.noise_scheduler.config.prediction_type 
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = state_traj
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction='none')
        loss = loss * action_mask.type(loss.dtype)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()
        return loss
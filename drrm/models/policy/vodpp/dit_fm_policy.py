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
from .vision.obs_encoder import SceneEncoder as VODPPlusEncoder
from .common.normalizer import LinearNormalizer
from .common.pytorch_util import dict_apply
from .common.module_attr_mixin import ModuleAttrMixin

import yaml
import json
from dataclasses import dataclass
from typing import Optional
from transformers import PretrainedConfig, PreTrainedModel

@dataclass
class VODPPlusDitFlowMatchingConfig(PretrainedConfig):
    shape_meta: dict
    noise_scheduler: dict
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
    self_attn_first: bool = True
    block_type: str = ''

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


class VODPPlusDitFlowMatching(BasePolicy, PreTrainedModel, ModuleAttrMixin):
    config_class = VODPPlusDitFlowMatchingConfig

    def __init__(self, config: VODPPlusDitFlowMatchingConfig):
        super().__init__(config)
        self.num_timestep_buckets = config.noise_scheduler['num_train_timesteps']
        self.num_inference_timesteps = config.noise_scheduler['num_inference_timesteps']
        self.beta_dist = torch.distributions.Beta(
            config.noise_scheduler['noise_beta_alpha'], 
            config.noise_scheduler['noise_beta_beta']
        )
        self.noise_s = config.noise_scheduler['noise_s']
        
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
        action_mask = torch.zeros(1, horizon, action_dim).bool()
        action_mask[:,n_obs_steps:,:] = True

        # create diffusion model
        num_patches = H * W
        scene_cond_len = (config.n_obs_steps * V * num_patches)
        scene_pos_embed_config = [("image", (config.n_obs_steps, V, num_patches)),]
        
        if config.block_type.lower() == 'invblock':
            from .diffusion.conditional_dit_decouple_head import DecoupleDiT
            self.model = DecoupleDiT(
                output_dim=action_dim,
                horizon=horizon,
                n_obs_steps=n_obs_steps,
                hidden_size=hidden_size,
                depth=config.depth,
                num_heads=config.num_heads,
                self_attn_first=config.self_attn_first,
                scene_cond_len=scene_cond_len,
                scene_pos_embed_config=scene_pos_embed_config
            )
        elif config.block_type.lower() == 'largeblock':
            from .diffusion.conditional_largedit_head import LargeDiT
            self.model = LargeDiT(
                output_dim=action_dim,
                horizon=horizon,
                n_obs_steps=n_obs_steps,
                hidden_size=hidden_size,
                depth=config.depth,
                num_heads=config.num_heads,
                action_only=False,
                scene_cond_len=scene_cond_len,
                scene_pos_embed_config=scene_pos_embed_config
            )
        elif config.block_type.lower() == 'nostateblock':
            from .diffusion.conditional_largedit_head import LargeDiT
            self.model = LargeDiT(
                output_dim=action_dim,
                horizon=horizon,
                n_obs_steps=n_obs_steps,
                hidden_size=hidden_size,
                depth=config.depth,
                num_heads=config.num_heads,
                action_only=True,
                scene_cond_len=scene_cond_len,
                scene_pos_embed_config=scene_pos_embed_config
            )
        elif config.block_type.lower() == 'vablock':
            from .diffusion.conditional_vadit_head import VADiT
            self.model = VADiT(
                output_dim=action_dim,
                horizon=horizon,
                n_obs_steps=n_obs_steps,
                hidden_size=hidden_size,
                depth=config.depth,
                num_heads=config.num_heads,
                self_attn_first=False,
                scene_cond_len=scene_cond_len,
                scene_pos_embed_config=scene_pos_embed_config
            )
        else:
            from .diffusion.conditional_dit_head import DiT
            self.model = DiT(
                output_dim=action_dim,
                horizon=horizon,
                hidden_size=hidden_size,
                depth=config.depth,
                num_heads=config.num_heads,
                self_attn_first=config.self_attn_first,
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
        self.action_mask = action_mask
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
            self, scene_cond, state_traj, action_mask, return_flow:bool = False, **kwargs
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
        flow = []
        if return_flow:
            flow.append(noisy_action.detach().cpu().numpy())

        # Set step values
        num_steps = self.num_inference_timesteps
        dt = 1.0 / num_steps

        state_mask = ~action_mask
        for t in range(num_steps):
            # timesteps = t.unsqueeze(-1).to(device)
            t_discretized = int(t / float(num_steps) * self.num_timestep_buckets)
            timesteps = torch.full(
                size=(batch_size,), fill_value=t_discretized, device=device
            )
            # Prepare state-action trajectory
            noisy_action[state_mask] = state_traj[state_mask]
            state_action_traj = self.state_adaptor(noisy_action)
            
            # Predict the model output
            pred_velocity = self.model(state_action_traj, timesteps, scene_cond)
            
            # Compute previous actions: x_t -> x_t-1
            noisy_action = noisy_action + dt * pred_velocity
            if return_flow:
                flow.append(noisy_action.detach().cpu().numpy())
            # noisy_action = noisy_action.to(state_traj.dtype)
        
        # Finally apply the action mask to mask invalid action dimensions
        noisy_action[state_mask] = state_traj[state_mask]

        return noisy_action, flow

    @torch.no_grad()
    def predict_action(self, obs_dict: Dict[str, torch.Tensor], return_flow:bool = False) -> Dict[str, torch.Tensor]:
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
        state_traj[:,:To,:] = this_nobs['agent_pos'].view(B,To,-1)
        action_mask = self.action_mask.expand(B, -1, -1).to(device=device)

        # run sampling
        nsample, flow = self.conditional_sample(scene_cond, state_traj, action_mask, return_flow)
        
        # unnormalize prediction
        naction_pred = nsample[...,:Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        # get action
        start = To
        end = start + self.n_action_steps
        action = action_pred[:,start:end]
        flow = [smp[:,start:end] for smp in flow]
        
        result = {
            'action': action,
            'action_pred': action_pred,
            'flow': flow,
            'gt': nobs['agent_pos'][:,start:end]
        }
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def sample_time(self, batch_size, device, dtype):
        sample = self.beta_dist.sample([batch_size]).to(device, dtype=dtype)
        return (self.noise_s - sample) / self.noise_s

    def compute_loss(self, batch):
        # normalize input
        assert 'valid_mask' not in batch
        nobs = self.normalizer.normalize(batch['obs'])
        actions = self.normalizer['action'].normalize(batch['action'])
        batch_size = actions.shape[0]
        horizon = actions.shape[1]

        # handle different ways of passing observation
        # reshape B, T, ... to B*T
        this_nobs = dict_apply(nobs, lambda x: x[:,:self.n_obs_steps,...].reshape(-1,*x.shape[2:]))
        nobs_features = self.obs_encoder(this_nobs) # (BS, VP, Do)
        scene_cond = self.adapt_conditions(nobs_features) # (BS, VP, hidden_size)
        scene_cond = scene_cond.view(batch_size, -1, scene_cond.shape[-1]) # (B, SVP, hidden_size)
        state_traj = actions
        action_mask = self.action_mask.expand(batch_size, -1, -1).to(device=state_traj.device)

        # Sample noise that we'll add to the images
        noise = torch.randn(actions.shape, device=actions.device)
        # Sample a random timestep for each image
        t = self.sample_time(
            actions.shape[0],
            device=actions.device,
            dtype=actions.dtype
        )
        t = t[:, None, None]  # shape (B,1,1) for broadcast
        # Convert (continuous) t -> discrete if needed
        timesteps = (t[:, 0, 0] * self.num_timestep_buckets).long() # shape (B,)
        
        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_traj = (1 - t) * noise + t * actions
        velocity = actions - noise
        
        # compute loss mask
        cond_mask = ~action_mask
        # apply conditioning
        noisy_traj[cond_mask] = state_traj[cond_mask]
        state_noisy_tokens = self.state_adaptor(noisy_traj)
        
        # Predict the noise residual
        pred = self.model(state_noisy_tokens, timesteps, scene_cond)

        loss = F.mse_loss(pred, velocity, reduction='none')
        loss = loss * action_mask.type(loss.dtype)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()
        return loss
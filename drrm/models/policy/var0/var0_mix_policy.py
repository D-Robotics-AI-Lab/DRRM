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
class VAR0MixConfig(PretrainedConfig):
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
    self_attn_first: bool = False
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


class VAR0Mix(BasePolicy, PreTrainedModel, ModuleAttrMixin):
    config_class = VAR0MixConfig

    def __init__(self, config: VAR0MixConfig):
        super().__init__(config)
        self.num_timestep_buckets = config.noise_scheduler['num_train_timesteps']
        # self.num_inference_timesteps = config.noise_scheduler['num_inference_timesteps']
        # self.beta_dist = torch.distributions.Beta(
        #     config.noise_scheduler['noise_beta_alpha'], 
        #     config.noise_scheduler['noise_beta_beta']
        # )
        # self.noise_s = config.noise_scheduler['noise_s']
        self.t_action = config.noise_scheduler['t_action']
        self.t_obs = config.noise_scheduler['t_obs']
        self.prediction_type = config.noise_scheduler['prediction_type']
        
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

        self.prompt = torch.nn.Embedding(
            num_embeddings=action_mask.sum(),
            embedding_dim=action_dim
        )

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
        act_0 = torch.zeros(
            size=(batch_size, self.horizon, self.action_dim), 
            dtype=dtype, device=device
        )
        

        state_mask = ~action_mask
        s = torch.zeros(batch_size, device=act_0.device, dtype=torch.long)
        t = (s + int(self.t_action * self.num_timestep_buckets)).detach() # shape (B,)

        # Prepare state-action trajectory
        act_0[state_mask] = state_traj[state_mask]
        # Predict the model output
        pred = self.model(self.state_adaptor(act_0), s, scene_cond)
        # Compute previous actions
        if self.prediction_type == 'velocity': # velocity
            act_t = act_0 + pred * self.t_action
        elif self.prediction_type == 'sample': # sample
            act_t = pred * self.t_action + act_0 * (1-self.t_action)
        else:
            act_t = pred
        
        # Prepare state-action trajectory
        act_t[state_mask] = state_traj[state_mask]
        # Predict the model output
        pred = self.model(self.state_adaptor(act_t), s, scene_cond)
        # Compute previous actions
        if self.prediction_type == 'velocity': # velocity
            act_1 = act_0 + pred * (1-self.t_action)
        elif self.prediction_type == 'sample': # sample
            act_1 = pred
        else:
            act_1 = pred

        return act_1

    @torch.no_grad()
    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
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

    # def sample_time(self, batch_size, device, dtype):
    #     sample = self.beta_dist.sample([batch_size]).to(device, dtype=dtype)
    #     return (self.noise_s - sample) / self.noise_s

    def compute_loss(self, batch):
        # normalize input
        assert 'valid_mask' not in batch
        nobs = self.normalizer.normalize(batch['obs'])
        act_1 = self.normalizer['action'].normalize(batch['action'])
        batch_size = act_1.shape[0]
        horizon = act_1.shape[1]

        # handle different ways of passing observation
        # reshape B, T, ... to B*T
        this_nobs = dict_apply(nobs, lambda x: x[:,:self.n_obs_steps,...].reshape(-1,*x.shape[2:]))
        nobs_features = self.obs_encoder(this_nobs) # (BS, VP, Do)
        obs_1 = self.adapt_conditions(nobs_features) # (BS, VP, hidden_size)
        action_mask = self.action_mask.expand(batch_size, -1, -1).to(device=act_1.device)

        # Sample noise for observations
        noise = torch.randn(obs_1.shape, device=obs_1.device)
        obs_t = obs_1 + (1 - self.t_obs) * noise # (BS, VP, hidden_size)

        # Sample noise for actions
        noise = torch.randn(act_1.shape, device=act_1.device)
        act_0 = act_1.clone().detach()
        gen_act_step = self.action_mask[...,0].sum()
        index_batch = torch.arange(
            gen_act_step, 
            dtype=torch.long, 
            device=act_1.device
        )
        prompt_batch = self.prompt(index_batch).unsqueeze(0).expand(batch_size, -1, -1)
        act_0[:,-gen_act_step:,:] = prompt_batch
        act_t = act_1 + (1 - self.t_action) * noise

        # Convert (continuous) t -> discrete
        s = torch.zeros(batch_size, device=act_1.device, dtype=torch.long)
        t = (s + int(self.t_action * self.num_timestep_buckets)).detach() # shape (B,)
        
        # Predict the noise residual
        pred_0 = self.model(self.state_adaptor(act_0), s, obs_t)
        pred_t = self.model(self.state_adaptor(act_t), t, obs_1)

        def mse_loss(pred, gt):
            loss = F.mse_loss(pred[action_mask], gt[action_mask])
            return loss

        if self.prediction_type == 'velocity': # velocity
            loss_0 = mse_loss(pred_0, act_1-act_0)
            loss_t = mse_loss(pred_t, act_1-act_0)
        elif self.prediction_type == 'sample': # sample
            loss_0 = mse_loss(pred_0, act_1)
            loss_t = mse_loss(pred_t, act_1)/(1-self.t_action)
        else: # hybrid
            loss_0 = mse_loss(pred_0, (act_1-act_0)*self.t_action)/self.t_action
            loss_t = mse_loss(pred_t, act_1)/(1-self.t_action)
        loss = loss_0 + loss_t
        return {
            'loss': loss, 
            'loss (step=0)': loss_0.detach(), 
            f'loss (step={self.t_action})': loss_t.detach()
        }
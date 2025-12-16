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
from .diffusion.conditional_dit_planner import DiTPlanner
from .emvis.solver import Solver

import yaml
import json
from dataclasses import dataclass
from typing import Optional
from transformers import PretrainedConfig, PreTrainedModel

@dataclass
class VAR0Config(PretrainedConfig):
    shape_meta: dict
    noise_scheduler: dict
    obs_encoder: VODPPlusEncoder
    scene_adaptor: str
    hidden_size: int
    depth: int
    num_heads: int
    horizon: int
    n_action_steps: int
    n_obs_steps: int
    # obs_as_global_cond: bool = True
    # diffusion_step_embed_dim: int = 256
    # down_dims: tuple = (256,512,1024)
    # kernel_size: int = 5
    # n_groups: int = 8
    # cond_predict_scale: bool = True
    self_attn_first: bool = True
    # block_type: str = ''
    solver_residual: bool = False

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

########################################################
# 1. DiT Flow matching Scenario Planner
# 2. IK Solver from Latent Scene Features
########################################################

class VAR0(BasePolicy, PreTrainedModel, ModuleAttrMixin):
    config_class = VAR0Config

    def __init__(self, config: VAR0Config):
        super().__init__(config)
        self.num_timestep_buckets = config.noise_scheduler['num_train_timesteps']
        self.num_inference_timesteps = config.noise_scheduler['num_inference_timesteps']
        self.beta_dist = torch.distributions.Beta(
            config.noise_scheduler['noise_beta_alpha'], 
            config.noise_scheduler['noise_beta_beta']
        )
        self.noise_s = config.noise_scheduler['noise_s']
        self.prediction_type = config.noise_scheduler.get('prediction_type', 'velocity')
        
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
        num_patches = H * W
        obs_feature_dim = dim
        action_mask = torch.zeros(1, horizon, action_dim).bool()
        action_mask[:,n_obs_steps:,:] = True
        cond_mask = torch.zeros(horizon, V * num_patches).bool()
        cond_mask[:n_obs_steps,:] = True
        cond_mask = cond_mask.view(-1)

        # self.scene_adaptor = self.build_condition_adapter(
        #     config.scene_adaptor, 
        #     in_features=obs_feature_dim, 
        #     out_features=hidden_size
        # )

        # create diffusion planner
        
        # gen_steps, cond_steps = horizon-config.n_obs_steps, config.n_obs_steps
        gen_steps, cond_steps = horizon, config.n_obs_steps
        scene_gen_len = gen_steps * V * num_patches
        scene_gen_pe_config = [("image", (gen_steps, V, num_patches)),]
        scene_cond_len = (cond_steps * V * num_patches)
        scene_cond_pe_config = [("image", (cond_steps, V, num_patches)),]

        self.planner = DiTPlanner(
            hidden_size=hidden_size,
            depth=config.depth,
            num_heads=config.num_heads,
            self_attn_first=config.self_attn_first,
            gen_len=scene_gen_len,
            gen_pe_config=scene_gen_pe_config,
            cond_len=scene_cond_len,
            cond_pe_config=scene_cond_pe_config
        )

        # create robot solver
        self.solver = Solver(
            hidden_size=hidden_size,
            action_dim=action_dim,
            shape_in=(H, W),
            residual=config.solver_residual,
        )

        self.normalizer = LinearNormalizer()
        # self.normalizer = None
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.gen_steps = gen_steps
        self.scene_gen_len = scene_gen_len
        self.cond_steps = cond_steps
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.action_mask = action_mask
        self.cond_mask = cond_mask
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
            self, cond, **kwargs
        ) -> torch.Tensor:
        '''
        scene_cond: image conditional data, (batch_size, patch_num, hidden_size).
        state_traj: (batch_size, 1, hidden_size), state trajectory.
        
        return: (batch_size, horizon, action_dim)
        '''
        device = cond.device
        dtype = cond.dtype
        batch_size = cond.shape[0]
        noisy_scene = torch.randn(
            size=(batch_size, self.scene_gen_len, self.obs_feature_dim), 
            dtype=dtype, device=device
        )

        # Set step values
        num_steps = self.num_inference_timesteps
        dt = 1.0 / num_steps

        for t in range(num_steps):
            # timesteps = t.unsqueeze(-1).to(device)
            t_discretized = int(t / float(num_steps) * self.num_timestep_buckets)
            timesteps = torch.full(
                size=(batch_size,), fill_value=t_discretized, device=device
            )
            if 'state' in self.prediction_type:
                pred = self.planner(noisy_scene, timesteps, cond)
                # pred_velocity = (pred - noisy_scene)/(1-t/num_steps)
                # pred_strid = (1.0 / num_steps) * pred_velocity
                pred_strid = (pred - noisy_scene) / (num_steps - t)
                noisy_scene = noisy_scene + pred_strid
            else:
                # Predict the model output
                pred_velocity = self.planner(noisy_scene, timesteps, cond)
                # Compute previous actions: x_t -> x_t-1
                noisy_scene = noisy_scene + dt * pred_velocity
                # noisy_scene = noisy_scene.to(state_traj.dtype)

        return noisy_scene

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
        To = self.cond_steps

        # build input
        device = self.device
        dtype = self.dtype
        cond_mask = self.cond_mask.to(device=device)

        # handle different ways of passing observation
        # condition through global feature
        this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
        nobs_features = self.obs_encoder(this_nobs) # (B*To, V*P, Do)

        # run sampling
        cond = nobs_features.view(B, -1, Do)
        scene_flow = self.conditional_sample(cond)
        
        # unnormalize prediction
        scene_flow = scene_flow.view(B * T, -1, Do)
        naction_pred = self.solver(scene_flow).view(B, T, Da)
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
        # this_nobs = dict_apply(nobs, lambda x: x[:,:self.n_obs_steps,...].reshape(-1,*x.shape[2:]))
        this_nobs = dict_apply(nobs, lambda x: x.reshape(-1,*x.shape[2:]))
        nobs_features = self.obs_encoder(this_nobs) # (BS, VP, Do)
        nobs_features = nobs_features.view(batch_size, -1, nobs_features.shape[-1]) # (B, SVP, Do)
        
        scene_flow = nobs_features
        cond_mask = self.cond_mask.to(device=scene_flow.device)
        x = scene_flow
        cond = scene_flow[:,cond_mask,...]
        
        # Sample noise that we'll add to the images
        noise = torch.randn(x.shape, device=x.device)
        # noise = torch.randn(actions.shape, device=actions.device)
        # Sample a random timestep for each image
        t = self.sample_time(
            batch_size,
            device=noise.device,
            dtype=noise.dtype
        )
        t = t[:, None, None]  # shape (B,1,1) for broadcast
        # Convert (continuous) t -> discrete if needed
        timesteps = (t[:, 0, 0] * self.num_timestep_buckets).long() # shape (B,)
        
        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_x = (1 - t) * noise + t * x
        
        dim = scene_flow.shape[-1]
        
        if 'state_gen' in self.prediction_type:
            # compute generated frames state loss
            pred_scene = self.planner(noisy_x, timesteps, cond) # (B, gen_len*V*P, hidden_size)
            gen_loss = F.kl_div(
                F.log_softmax(pred_scene, dim=-1), 
                F.softmax(x, dim=-1), reduction='none'
            )
            gen_loss = gen_loss.sum(-1).mean()
            gen_loss_log = gen_loss.detach()
        elif 'state' in self.prediction_type:
            # compute generated frames state loss
            pred_scene = self.planner(noisy_x, timesteps, cond) # (B, gen_len*V*P, hidden_size)
            gen_loss_log = F.kl_div(
                F.log_softmax(pred_scene, dim=-1), 
                F.softmax(x, dim=-1), reduction='none'
            )
            gen_loss_log = gen_loss_log.sum(-1).mean().detach()
            gen_loss = 0
        elif 'velocity_gen' in self.prediction_type:
            # compute generated frames velocity loss
            velocity = x - noise
            pred_velocity = self.planner(noisy_x, timesteps, cond) # (B, gen_len*V*P, hidden_size)
            pred_scene = noise + pred_velocity
            if 'mse' in self.prediction_type:
                gen_loss = F.mse_loss(pred_velocity, velocity)
            else:
                gen_loss = F.kl_div(
                    F.log_softmax(pred_velocity, dim=-1), 
                    F.softmax(velocity, dim=-1), reduction='none'
                )
            gen_loss = gen_loss.sum(-1).mean()
            gen_loss_log = gen_loss.detach()
        else: 
            velocity = x - noise
            # compute generated frames velocity loss
            pred_velocity = self.planner(noisy_x, timesteps, cond) # (B, gen_len*V*P, hidden_size)
            pred_scene = noise + pred_velocity
            if 'mse' in self.prediction_type:
                gen_loss_log = F.mse_loss(pred_velocity, velocity)
            else:
                gen_loss_log = F.kl_div(
                    F.log_softmax(pred_velocity, dim=-1), 
                    F.softmax(velocity, dim=-1), reduction='none'
                )
            gen_loss_log = gen_loss.sum(-1).mean()
            gen_loss_log = gen_loss_log.sum(-1).mean().detach()
            gen_loss = 0

        # compute generated frames inverse loss
        pred_scene = pred_scene.reshape(batch_size * self.gen_steps, -1, dim)
        pred = self.solver(pred_scene).view(batch_size, self.gen_steps, -1)
        gen_inv_loss = F.mse_loss(pred, actions)

        # compute directly inverse loss
        
        scene_flow = scene_flow.view(batch_size * horizon, -1, dim)
        pred = self.solver(scene_flow).view(batch_size, horizon, -1)
        inv_loss = F.mse_loss(pred, actions)
        inv_loss_log = inv_loss.detach()
        if 'inv' not in self.prediction_type:
            inv_loss = torch.tensor(0.0, device=actions.device)
        
        if 'balancing' in self.prediction_type:
            constrain = (inv_loss + gen_loss).detach()
            if constrain < 1e-4:
                constrain = torch.tensor(1e-4, device=actions.device)
            beta = (gen_inv_loss / constrain * 0.1).detach()
            loss = beta * (inv_loss + gen_loss) + gen_inv_loss
        else:
            loss = inv_loss + gen_inv_loss + gen_loss
        return {'loss': loss, 'inv_loss': inv_loss_log, 'gen_inv_loss': gen_inv_loss, 'gen_loss': gen_loss_log}
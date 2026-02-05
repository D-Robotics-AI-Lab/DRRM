import torch
import os
import numpy as np
import hydra
from pathlib import Path
from collections import deque

import yaml
from datetime import datetime
import importlib
import dill
from argparse import ArgumentParser
from drrm.common.pytorch_util import dict_apply
from drrm.models.base_policy import BasePolicy
from drrm.models.base_runner import BaseRunner, load_policy


class VODPPRunner(BaseRunner):

    def __init__(
        self,
        policy: BasePolicy = None,
        checkpoint_dir: str = None,
        mixed_precision: str = 'bf16',
        output_dir: str = None,
        eval_episodes: int = 20,
        max_steps: int = 300,
        # n_obs_steps = 3,
        # n_action_steps = 8,
        fps: int = 10,
        crf: int = 22,
        tqdm_interval_sec: float = 5.0,
        task_name: str = None,
        **kwargs,
    ):
        super().__init__(output_dir)
        if policy is not None:
            self.policy = policy
        else:
            assert checkpoint_dir is not None, "Either policy or checkpoint_dir must be provided"
            self.policy = load_policy(checkpoint_dir, use_ckp_code=False)
            self.policy.eval()
            self.policy.to('cuda') if 'device' not in kwargs else self.policy.to(kwargs['device'])
        if mixed_precision == 'bf16':
            self.dtype = torch.bfloat16
        elif mixed_precision == 'fp16':
            self.dtype = torch.float16
        else:
            self.dtype = torch.float32

        self.task_name = task_name
        self.eval_episodes = eval_episodes
        self.fps = fps
        self.crf = crf
        self.n_obs_steps = self.policy.n_obs_steps
        self.n_action_steps = self.policy.n_action_steps
        self.max_steps = max_steps
        self.tqdm_interval_sec = tqdm_interval_sec

        self.obs = deque(maxlen=self.n_obs_steps + 1)
        self.env = None

    def stack_last_n_obs(self, all_obs, n_steps):
        assert len(all_obs) > 0
        all_obs = list(all_obs)
        if isinstance(all_obs[0], np.ndarray):
            result = np.zeros((n_steps, ) + all_obs[-1].shape, dtype=all_obs[-1].dtype)
            start_idx = -min(n_steps, len(all_obs))
            result[start_idx:] = np.array(all_obs[start_idx:])
            if n_steps > len(all_obs):
                # pad
                result[:start_idx] = result[start_idx]
        elif isinstance(all_obs[0], torch.Tensor):
            result = torch.zeros((n_steps, ) + all_obs[-1].shape, dtype=all_obs[-1].dtype)
            start_idx = -min(n_steps, len(all_obs))
            result[start_idx:] = torch.stack(all_obs[start_idx:])
            if n_steps > len(all_obs):
                # pad
                result[:start_idx] = result[start_idx]
        else:
            raise RuntimeError(f"Unsupported obs type {type(all_obs[0])}")
        return result

    def reset_obs(self):
        self.obs.clear()

    def update_obs(self, current_obs):
        self.obs.append(current_obs)

    def get_n_steps_obs(self):
        assert len(self.obs) > 0, "no observation is recorded, please update obs first"

        result = dict()
        for key in self.obs[0].keys():
            result[key] = self.stack_last_n_obs([obs[key] for obs in self.obs], self.n_obs_steps)

        return result

    def get_action(self, observaton=None):
        policy: BasePolicy = self.policy
        if observaton != None:
            self.obs.append(observaton)  # update
        device, dtype = policy.device, self.dtype
        obs = self.get_n_steps_obs()

        # create obs dict
        np_obs_dict = dict(obs)
        # device transfer
        obs_dict = dict_apply(np_obs_dict, lambda x: torch.from_numpy(x).to(device=device))
        # run policy
        with torch.no_grad():
            obs_dict_input = {}  # flush unused keys
            obs_dict_input["head_cam"] = obs_dict["head_cam"].unsqueeze(0)
            obs_dict_input['front_cam'] = obs_dict['front_cam'].unsqueeze(0)
            obs_dict_input["left_cam"] = obs_dict["left_cam"].unsqueeze(0)
            obs_dict_input["right_cam"] = obs_dict["right_cam"].unsqueeze(0)
            obs_dict_input["agent_pos"] = obs_dict["agent_pos"].unsqueeze(0)

            device_type = str(device)
            if dtype == torch.float32:
                action_dict = policy.predict_action(obs_dict_input)
            else:
                with torch.autocast(device_type=device_type, dtype=dtype):
                    action_dict = policy.predict_action(obs_dict_input)

        # device_transfer
        np_action_dict = action_dict["action"].detach().to("cpu").numpy()
        action = np_action_dict.squeeze(0)[:self.n_action_steps]
        return action
    
    def get_last_obs(self):
        return self.obs[-1]

    def run(self, policy: BasePolicy):
        pass

if __name__ == '__main__':
    test = VODPPRunner('./')
    print('ready')
import os
import numpy as np
from drrm.models.policy.vodpp.env_runner.base_runner import VODPPRunner
import yaml

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))

def encode_obs(observation):
    head_cam = (np.moveaxis(observation["observation"]["head_camera"]["rgb"], -1, 0) / 255)
    left_cam = (np.moveaxis(observation["observation"]["left_camera"]["rgb"], -1, 0) / 255)
    right_cam = (np.moveaxis(observation["observation"]["right_camera"]["rgb"], -1, 0) / 255)
    obs = dict(
        head_cam=head_cam,
        left_cam=left_cam,
        right_cam=right_cam,
    )
    obs["agent_pos"] = observation["joint_action"]["vector"]
    return obs


def get_model(usr_args):
    if usr_args['checkpoint_num'] is None:
        ckpt_file = f"{ROOT_DIR}/checkpoints/vodpp_{usr_args['task_name']}_{usr_args['expert_data_num']}_{usr_args['ckpt_setting']}"
    else:
        ckpt_file = f"{ROOT_DIR}/checkpoints/vodpp_{usr_args['task_name']}_{usr_args['expert_data_num']}_{usr_args['ckpt_setting']}/checkpoint-{usr_args['checkpoint_num']}"
    kwargs = {}
    if 'device' in usr_args:
        kwargs['device'] = f"cuda:{usr_args['device']}"
    return VODPPRunner(checkpoint_dir=ckpt_file, **kwargs)


def eval(TASK_ENV, model, observation):
    """
    TASK_ENV: Task Environment Class, you can use this class to interact with the environment
    model: The model from 'get_model()' function
    observation: The observation about the environment
    """
    obs = encode_obs(observation)
    instruction = TASK_ENV.get_instruction()
    # if all([(model.obs[-1][k] == obs[k]).all() for k in model.obs[-1]]):
    #     print("Warning: Redundant Observation Detected!")

    # ======== Get Action ========
    actions = model.get_action(obs)

    for action in actions:
        TASK_ENV.take_action(action)
        observation = TASK_ENV.get_obs()
        obs = encode_obs(observation)
        model.update_obs(obs)

def reset_model(model):
    model.reset_obs()

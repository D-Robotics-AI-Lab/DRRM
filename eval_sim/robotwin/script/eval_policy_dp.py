import sys
import torch  
import os
import numpy as np
import hydra
from pathlib import Path
from collections import deque
import traceback
from copy import deepcopy
import multiprocessing as mp
import yaml
from datetime import datetime
import importlib
import argparse
from omegaconf import OmegaConf
from safetensors.torch import load_model
import time

from drrm.common.pytorch_util import dict_apply

# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)



class DPRunner:
    def __init__(self,
                 output_dir,
                 eval_episodes=20,
                 max_steps=300,
                 n_obs_steps=3,
                 n_action_steps=8,
                 fps=10,
                 crf=22,
                 tqdm_interval_sec=5.0,
                 task_name=None,
    ):
        self.task_name = task_name
        self.eval_episodes = eval_episodes
        self.fps = fps
        self.crf = crf
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.max_steps = max_steps
        self.tqdm_interval_sec = tqdm_interval_sec

        self.obs = deque(maxlen=n_obs_steps+1)
        self.env = None

    def stack_last_n_obs(self, all_obs, n_steps):
        assert(len(all_obs) > 0)
        all_obs = list(all_obs)
        if isinstance(all_obs[0], np.ndarray):
            result = np.zeros((n_steps,) + all_obs[-1].shape, 
                dtype=all_obs[-1].dtype)
            start_idx = -min(n_steps, len(all_obs))
            result[start_idx:] = np.array(all_obs[start_idx:])
            if n_steps > len(all_obs):
                # pad
                result[:start_idx] = result[start_idx]
        elif isinstance(all_obs[0], torch.Tensor):
            result = torch.zeros((n_steps,) + all_obs[-1].shape, 
                dtype=all_obs[-1].dtype)
            start_idx = -min(n_steps, len(all_obs))
            result[start_idx:] = torch.stack(all_obs[start_idx:])
            if n_steps > len(all_obs):
                # pad
                result[:start_idx] = result[start_idx]
        else:
            raise RuntimeError(f'Unsupported obs type {type(all_obs[0])}')
        return result
    
    def reset_obs(self):
        self.obs.clear()

    def update_obs(self, current_obs):
        self.obs.append(current_obs)

    def get_n_steps_obs(self):
        assert(len(self.obs) > 0), 'no observation is recorded, please update obs first'

        result = dict()
        for key in self.obs[0].keys():
            result[key] = self.stack_last_n_obs(
                [obs[key] for obs in self.obs],
                self.n_obs_steps
            )

        return result

    def get_action(self, policy, observaton=None):
        device, dtype = policy.device, policy.dtype
        if observaton is not None:
            self.obs.append(observaton) # update
        obs = self.get_n_steps_obs()

        # create obs dict
        np_obs_dict = dict(obs)
        # device transfer
        obs_dict = dict_apply(np_obs_dict, lambda x: torch.from_numpy(x).to(device=device))
        # run policy
        with torch.no_grad():
            obs_dict_input = {}  # flush unused keys
            obs_dict_input['head_cam'] = obs_dict['head_cam'].unsqueeze(0)
            # obs_dict_input['front_cam'] = obs_dict['front_cam'].unsqueeze(0)
            # obs_dict_input['left_cam'] = obs_dict['left_cam'].unsqueeze(0)
            # obs_dict_input['right_cam'] = obs_dict['right_cam'].unsqueeze(0)
            obs_dict_input['agent_pos'] = obs_dict['agent_pos'].unsqueeze(0)
            
            action_dict = policy.predict_action(obs_dict_input)

        # device_transfer
        np_action_dict = dict_apply(action_dict, lambda x: x.detach().to('cpu').numpy())
        action = np_action_dict['action'].squeeze(0)
        return action

class DP:
    def __init__(self, cfg: OmegaConf):
        model_cfg = OmegaConf.load(cfg.config_name)
        
        # 如果配置文件中有defaults字段，需要手动处理继承
        if 'defaults' in model_cfg:
            base_config_path = os.path.join(os.path.dirname(cfg.config_name), model_cfg.defaults[0])
            if not os.path.exists(base_config_path):
                base_config_path = base_config_path + '.yaml'
            base_cfg = OmegaConf.load(base_config_path)
            # 合并配置，model_cfg会覆盖base_cfg中的同名配置
            model_cfg = OmegaConf.merge(base_cfg, model_cfg)
        
        self.policy = hydra.utils.instantiate(model_cfg.model)
        load_model(self.policy, os.path.join(cfg.checkpoint_dir, "model.safetensors"), strict=False)    # TODO: strict=False
        self.policy.eval()
        self.policy.to('cuda')

        self.runner = DPRunner(output_dir=None)

    def update_obs(self, observation):
        self.runner.update_obs(observation)
    
    def get_action(self, observation=None):
        action = self.runner.get_action(self.policy, observation)
        return action

    def get_last_obs(self):
        return self.runner.obs[-1]

def test_policy_worker(task_name, args_copy, st_seed_list_sub, test_num_list_sub, gpu_id = None):
    if gpu_id != None: os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    Demo_class_copy = class_decorator(task_name)
    dp_copy = DP(args_copy)

    expert_check = True
    Demo_class_copy.suc = 0
    Demo_class_copy.test_num = test_num_list_sub[0]
    
    for now_seed, test_num in zip(st_seed_list_sub, test_num_list_sub):
        render_freq = args_copy['render_freq']
        args_copy['render_freq'] = 0
        if expert_check:
            Demo_class_copy.setup_demo(now_ep_num=test_num, seed = now_seed, is_test = True, ** args_copy)
            Demo_class_copy.play_once()
            Demo_class_copy.close()

        args_copy['render_freq'] = render_freq

        Demo_class_copy.setup_demo(now_ep_num=test_num, seed = now_seed, is_test = True, ** args_copy)
        Demo_class_copy.apply_dp(dp_copy, args_copy)
        Demo_class_copy.close()
        if Demo_class_copy.render_freq:
            Demo_class_copy.viewer.close()
        dp_copy.runner.reset_obs()
    
    return Demo_class_copy.suc
def test_policy(task_name, args, st_seed, test_num=20, num_process=1):
    expert_check = True
    print("Task name: ", args["task_name"])

    if num_process > 1:
        # 把test_num平均分配给num_process个进程
        test_num_list = range(test_num)
        test_num_list = np.array_split(test_num_list, num_process)

        # 拷贝Demo_class
        # Demo_class_list = [deepcopy(Demo_class) for _ in range(num_process)]

        # 拷贝args
        args_list = [deepcopy(args) for _ in range(num_process)]

        # 拷贝dp
        # dp_list = [deepcopy(dp) for _ in range(num_process)]
        # for ii, in enumerate(dp_list):
        #     ii.policy.to(f'cuda:{num_process % torch.cuda.device_count()}')
        
        # 设置每个进程的st_seed
        st_seed_list = np.array_split(range(st_seed, st_seed + test_num), num_process)


        # 进程池
        # args_list_zip = list(zip(Demo_class_list, args_list, dp_list, st_seed_list, test_num_list))
        # To use CUDA with multiprocessing, you must use the 'spawn' start method
        mp.set_start_method('spawn', force=True)
        processes = []
        gpu_num = torch.cuda.device_count()
        for i in range(num_process):
            p = mp.Process(target=test_policy_worker, args=(task_name, args_list[i], st_seed_list[i], test_num_list[i], i%gpu_num))
            # p = mp.Process(target=test_policy_worker, args=(Demo_class_list[i], args_list[i], dp_list[i], st_seed_list[i], test_num_list[i]))
            processes.append(p)
            p.start()
        for p in processes:
            p.join()

        # 合并结果
        return 0, len([f for f in os.listdir(args.save_dir) if f.endswith("success.mp4")])

    dp = DP(args)
    Demo_class = class_decorator(args['task_name'])

    Demo_class.suc = 0
    Demo_class.test_num =0

    now_id = 0
    succ_seed = 0
    suc_test_seed_list = []
    

    now_seed = st_seed
    while succ_seed < test_num:
        render_freq = args['render_freq']
        args['render_freq'] = 0
        
        if expert_check:
            try:
                Demo_class.setup_demo(now_ep_num=now_id, seed = now_seed, is_test = True, ** args)
                Demo_class.play_once()
                Demo_class.close()
            except Exception as e:
                stack_trace = traceback.format_exc()
                print(' -------------')
                print('Error: ', stack_trace)
                print(' -------------')
                Demo_class.close()
                now_seed += 1
                args['render_freq'] = render_freq
                print('error occurs !')
                continue

        if (not expert_check) or ( Demo_class.plan_success and Demo_class.check_success() ):
            succ_seed +=1
            suc_test_seed_list.append(now_seed)
        else:
            now_seed += 1
            args['render_freq'] = render_freq
            continue


        args['render_freq'] = render_freq

        Demo_class.setup_demo(now_ep_num=now_id, seed = now_seed, is_test = True, ** args)
        Demo_class.apply_dp(dp, args)

        now_id += 1
        Demo_class.close()
        if Demo_class.render_freq:
            Demo_class.viewer.close()
        dp.runner.reset_obs()
        print(f"{task_name} success rate: {Demo_class.suc}/{Demo_class.test_num}, current seed: {now_seed}\n")
        Demo_class._take_picture()
        now_seed += 1

    return now_seed, Demo_class.suc

def get_camera_config(camera_type):
    camera_config_path = Path(parent_directory).parent / 'task_config' / '_camera_config.yml'

    assert os.path.isfile(camera_config_path), "task config file is missing"

    with open(camera_config_path, 'r', encoding='utf-8') as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    assert camera_type in args, f'camera {camera_type} is not defined'
    return args[camera_type]

def class_decorator(task_name):
    sys.path.append(str(Path(parent_directory).parent))
    envs_module = importlib.import_module(f'envs.{task_name}')
    try:
        env_class = getattr(envs_module, task_name)
        env_instance = env_class()
    except:
        raise SystemExit("No Task")
    return env_instance

def main(args):
    with open(Path(parent_directory).parent / 'task_config' / (args.task_name + '.yml'), 'r', encoding='utf-8') as f:
        cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
    
    cfg['head_camera_type'] = args.head_camera_type
    head_camera_config = get_camera_config(cfg['head_camera_type'])
    cfg['head_camera_fovy'] = head_camera_config['fovy']
    cfg['head_camera_w'] = head_camera_config['w']
    cfg['head_camera_h'] = head_camera_config['h']
    head_camera_config = 'fovy' + str(cfg['head_camera_fovy']) + '_w' + str(cfg['head_camera_w']) + '_h' + str(cfg['head_camera_h'])
    
    cfg['wrist_camera_type'] = args.wrist_camera_type
    wrist_camera_config = get_camera_config(cfg['wrist_camera_type'])
    cfg['wrist_camera_fovy'] = wrist_camera_config['fovy']
    cfg['wrist_camera_w'] = wrist_camera_config['w']
    cfg['wrist_camera_h'] = wrist_camera_config['h']
    wrist_camera_config = 'fovy' + str(cfg['wrist_camera_fovy']) + '_w' + str(cfg['wrist_camera_w']) + '_h' + str(cfg['wrist_camera_h'])

    cfg['front_camera_type'] = args.front_camera_type
    front_camera_config = get_camera_config(cfg['front_camera_type'])
    cfg['front_camera_fovy'] = front_camera_config['fovy']
    cfg['front_camera_w'] = front_camera_config['w']
    cfg['front_camera_h'] = front_camera_config['h']
    front_camera_config = 'fovy' + str(cfg['front_camera_fovy']) + '_w' + str(cfg['front_camera_w']) + '_h' + str(cfg['front_camera_h'])

    # output camera config
    print('============= Camera Config =============\n')
    print('Head Camera Config:\n    type: '+ str(cfg['head_camera_type']) + '\n    fovy: ' + str(cfg['head_camera_fovy']) + '\n    camera_w: ' + str(cfg['head_camera_w']) + '\n    camera_h: ' + str(cfg['head_camera_h']))
    print('Wrist Camera Config:\n    type: '+ str(cfg['wrist_camera_type']) + '\n    fovy: ' + str(cfg['wrist_camera_fovy']) + '\n    camera_w: ' + str(cfg['wrist_camera_w']) + '\n    camera_h: ' + str(cfg['wrist_camera_h']))
    print('Front Camera Config:\n    type: '+ str(cfg['front_camera_type']) + '\n    fovy: ' + str(cfg['front_camera_fovy']) + '\n    camera_w: ' + str(cfg['front_camera_w']) + '\n    camera_h: ' + str(cfg['front_camera_h']))
    print('\n=======================================')

    cfg['expert_seed'] = args.seed
    cfg['checkpoint_dir'] = args.checkpoint_dir
    cfg['task_name'] = args.task_name
    cfg['config_name'] = args.config_name
    cfg['save_dir'] = args.save_dir
    cfg['num_process'] = args.num_process
    cfg = OmegaConf.create(cfg)

    # task = class_decorator(cfg['task_name'])

    st_seed = 100000 * (1+cfg['expert_seed'])
    suc_nums = []
    test_num = 100 
    topk = 1

    # dp = DP(cfg)

    # st_seed, suc_num = test_policy(cfg.task_name, task, cfg, dp, st_seed, test_num=test_num, num_process=cfg.num_process)
    st_seed, suc_num = test_policy(cfg.task_name, cfg, st_seed, test_num=test_num, num_process=cfg.num_process)
    suc_nums.append(suc_num)

    file_path = Path(cfg['save_dir']) / f'result.txt'
    with open(file_path, 'w') as file:
        file.write(f'Task Name: {cfg.task_name}\n')
        file.write(f"current time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}\n")
        file.write(f'Success Rate: {np.sum(suc_nums) / test_num}\n')
    print(f'Data has been saved to {file_path}')



if __name__ == "__main__":
    from test_render import Sapien_TEST
    Sapien_TEST()

    parser = argparse.ArgumentParser()
    parser.add_argument('--config-name', type=str, default='dp_baseline.yaml', help='config name to load')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints/dp_baseline/checkpoint-20000', help='checkpoint dir')
    parser.add_argument('--save-dir', type=str, default='eval_result/dp', help='save dir')
    parser.add_argument('--task-name', type=str, default='dual_bottles_pick_easy', help='task name')
    parser.add_argument('--head-camera-type', type=str, default='D435', help='head camera type')
    parser.add_argument('--wrist-camera-type', type=str, default='D435', help='wrist camera type')
    parser.add_argument('--front-camera-type', type=str, default='D435', help='front camera type')
    parser.add_argument('--seed', type=int, default=0, help='seed')
    parser.add_argument('--num-process', type=int, default=1, help='number of process')
    args = parser.parse_args()

    main(args)


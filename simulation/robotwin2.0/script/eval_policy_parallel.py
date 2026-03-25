from functools import partial
import sys
import os
os.environ["PYTHONWARNINGS"] = "ignore"
import subprocess
import time

import torch

sys.path.append("./")
sys.path.append(f"./policy")
sys.path.append("./description/utils")
from envs import CONFIGS_PATH
from envs.utils.create_actor import UnStableError

import numpy as np
from pathlib import Path
from collections import deque
import traceback

import yaml
from datetime import datetime
import importlib
import argparse
import multiprocessing as mp
from multiprocessing import Manager, Process, Queue
from video_process import merge_episodes
from description.utils.generate_episode_instructions import *


current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)
# os.chdir(os.path.join(os.getcwd(), "simulation/robotwin2.0"))

import subprocess
import os
import tempfile

def get_visible_gpu_num():
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        return [
            int(gpu.strip())
            for gpu in os.environ["CUDA_VISIBLE_DEVICES"].split(',')
        ]
    else:
        return [i for i in range(torch.cuda.device_count())]

def format_result(key: int, res: dict):
    s = f"【{key:03d}】"
    for k in  [
        'seed', 'success', 'time', 'fps', 
        'action_cnt', 'limit', 'action_gen_freq', 
        'infer_cnt', 'infer_freq',
        'start', 'end', 'count', 'pid', 'device'
    ]:
        if not k in res: continue
        elif k == 'time' or k == 'infer_time': 
            s += f"{k}: {int(res[k]):03d} s, "
        elif k in ['start', 'end']:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(res[k]))
            s += f"{k}: {timestamp}, "
        elif k == "fps" or "freq" in k:
            s += f"{k}: {res[k]:03.2f}, "
        elif k == 'success':
            if res[k]:
                result = 'Success'
            elif res[k]==None:
                result = 'None   '
            else:
                result = 'Fail   '
            s += f"result: {result}, "
        else:
            s += f"{k}: {res[k]}, "
    return s

def log_result(file_path, ind: int, result: dict, lock, **kwargs):
    with lock:
        result.update(**kwargs)
        with open(file_path, 'r', newline='') as f:
            lines = f.readlines()
        if ind >= len(lines):
            lines += ['\n']*(ind+1-len(lines))
        string = format_result(ind, result)
        lines[ind] = string + '\n'
        with open(file_path, 'w', newline='') as f:
            f.writelines(lines)

def class_decorator(task_name):
    envs_module = importlib.import_module(f"envs.{task_name}")
    try:
        env_class = getattr(envs_module, task_name)
        env_instance = env_class()
    except:
        raise SystemExit("No Task")
    return env_instance

def eval_function_decorator(policy_name, model_name):
    try:
        policy_model = importlib.import_module(policy_name)
        return getattr(policy_model, model_name)
    except ImportError as e:
        raise e

def get_camera_config(camera_type):
    camera_config_path = os.path.join(parent_directory, "../task_config/_camera_config.yml")

    assert os.path.isfile(camera_config_path), "task config file is missing"

    with open(camera_config_path, "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    assert camera_type in args, f"camera {camera_type} is not defined"
    return args[camera_type]

def get_embodiment_config(robot_file):
    robot_config_file = os.path.join(robot_file, "config.yml")
    with open(robot_config_file, "r", encoding="utf-8") as f:
        embodiment_args = yaml.load(f.read(), Loader=yaml.FullLoader)
    return embodiment_args

def main(usr_args):
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    task_name = usr_args["task_name"]
    task_config = usr_args["task_config"]
    ckpt_setting = usr_args["ckpt_setting"]
    checkpoint_num = usr_args.get("checkpoint_num", None)
    policy_name = usr_args["policy_name"]
    instruction_type = usr_args["instruction_type"]
    save_dir = None
    video_save_dir = None
    video_size = None
    demos = usr_args.get('expert_data_num', None)

    # get_model = eval_function_decorator(policy_name, "get_model")

    with open(f"./task_config/{task_config}.yml", "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    args['task_name'] = task_name
    args["task_config"] = task_config
    args["ckpt_setting"] = ckpt_setting
    args["checkpoint_num"] = checkpoint_num

    embodiment_type = args.get("embodiment")
    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")

    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        _embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def get_embodiment_file(embodiment_type):
        robot_file = _embodiment_types[embodiment_type]["file_path"]
        if robot_file is None:
            raise "No embodiment files"
        return robot_file

    with open(CONFIGS_PATH + "_camera_config.yml", "r", encoding="utf-8") as f:
        _camera_config = yaml.load(f.read(), Loader=yaml.FullLoader)

    head_camera_type = args["camera"]["head_camera_type"]
    args["head_camera_h"] = _camera_config[head_camera_type]["h"]
    args["head_camera_w"] = _camera_config[head_camera_type]["w"]

    if len(embodiment_type) == 1:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[1])
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False
    else:
        raise "embodiment items should be 1 or 3"

    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])

    if len(embodiment_type) == 1:
        embodiment_name = str(embodiment_type[0])
    else:
        embodiment_name = str(embodiment_type[0]) + "+" + str(embodiment_type[1])

    save_dir = Path(f"eval_result/{task_name}/{policy_name}/{task_config}/{ckpt_setting}-{checkpoint_num}-{demos}/{current_time}")
    save_dir.mkdir(parents=True, exist_ok=True)

    if args["eval_video_log"]:
        video_save_dir = save_dir
        camera_config = get_camera_config(args["camera"]["head_camera_type"])
        video_size = str(camera_config["w"]) + "x" + str(camera_config["h"])
        video_save_dir.mkdir(parents=True, exist_ok=True)
        args["eval_video_save_dir"] = video_save_dir

    # output camera config
    print("============= Config =============\n")
    print("\033[95mMessy Table:\033[0m " + str(args["domain_randomization"]["cluttered_table"]))
    print("\033[95mRandom Background:\033[0m " + str(args["domain_randomization"]["random_background"]))
    if args["domain_randomization"]["random_background"]:
        print(" - Clean Background Rate: " + str(args["domain_randomization"]["clean_background_rate"]))
    print("\033[95mRandom Light:\033[0m " + str(args["domain_randomization"]["random_light"]))
    if args["domain_randomization"]["random_light"]:
        print(" - Crazy Random Light Rate: " + str(args["domain_randomization"]["crazy_random_light_rate"]))
    print("\033[95mRandom Table Height:\033[0m " + str(args["domain_randomization"]["random_table_height"]))
    print("\033[95mRandom Head Camera Distance:\033[0m " + str(args["domain_randomization"]["random_head_camera_dis"]))

    print("\033[94mHead Camera Config:\033[0m " + str(args["camera"]["head_camera_type"]) + f", " +
          str(args["camera"]["collect_head_camera"]))
    print("\033[94mWrist Camera Config:\033[0m " + str(args["camera"]["wrist_camera_type"]) + f", " +
          str(args["camera"]["collect_wrist_camera"]))
    print("\033[94mEmbodiment Config:\033[0m " + embodiment_name)
    print("\n==================================")

    # TASK_ENV = class_decorator(args["task_name"])
    # get_env = partial(class_decorator, args["task_name"])
    args["policy_name"] = policy_name
    usr_args["left_arm_dim"] = len(args["left_embodiment_config"]["arm_joints_name"][0])
    usr_args["right_arm_dim"] = len(args["right_embodiment_config"]["arm_joints_name"][1])

    seed = usr_args["seed"]

    st_seed = 100000 * (1 + seed)
    suc_nums = []
    test_num = 100
    topk = 1

    # get_policy = partial(get_model, usr_args)
    st_seed, suc_num = eval_policy(
        task_name,
        args, usr_args,
        st_seed, test_num=test_num,
        num_process=usr_args.get("num_process", None),
        video_size=video_size,
        instruction_type=instruction_type
    )
    suc_nums.append(suc_num)

    topk_success_rate = sorted(suc_nums, reverse=True)[:topk]

    # file_path = os.path.join(save_dir, f"_result.txt")
    # with open(file_path, "w") as file:
    #     file.write(f"Timestamp: {current_time}\n\n")
    #     file.write(f"Instruction Type: {instruction_type}\n\n")
    #     # file.write(str(task_reward) + '\n')
    #     file.write("\n".join(map(str, np.array(suc_nums) / test_num)))

    # print(f"Data has been saved to {file_path}")
    # return task_reward
    merge_episodes(save_dir, "0_merged_episodes.mp4", speed=4.0)

def eval_policy(
    task_name,
    args, model_args,
    st_seed, test_num=100,
    num_process=None,
    video_size=None,
    instruction_type=None
):
    print(f"\033[34mTask Name: {args['task_name']}\033[0m")
    print(f"\033[34mPolicy Name: {args['policy_name']}\033[0m")
    # global model
    # get_model = eval_function_decorator(args["policy_name"], "get_model")
    # model = get_model(model_args)
    # model.policy = model.policy.cpu()
    # model.policy.share_memory()

    log_path = os.path.join(args["eval_video_save_dir"], f"_eval_log.txt")
    with open(log_path, 'w') as file:
        file.write(
            f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}"+
            '\n' * test_num
        )
    with Manager() as manager:
        seed = manager.Value('i', st_seed)
        need = manager.Value('i', test_num)
        lock = manager.Lock() 
        log_lock = manager.Lock()
        succ_cnt = manager.Value('i', 0, lock=True)
        finish_cnt = manager.Value('i', 0, lock=True)

        mp.set_start_method('spawn', force=True)
        processes = []
        results = []
        return_queue = Queue()
        gpu_list = get_visible_gpu_num()
        gpu_num = len(gpu_list)
        if num_process is None:
            num_process = gpu_num
        for i in range(num_process):
            gpu_id = gpu_list[i % gpu_num]
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            p = Process(
                target=eval_policy_worker, 
                args=(
                    task_name,
                    args, model_args,
                    seed, need, lock, succ_cnt, finish_cnt,
                    log_lock, log_path, return_queue,
                    test_num, video_size,
                    instruction_type, gpu_id
                )
            )
            processes.append(p)
            p.start()
        # for p in processes:
        #     p.join()
        # while not return_queue.empty():
        for i in range(num_process):
            results.append(return_queue.get())
            ind = list(results[-1].keys())[0]
            res_len = len(results[-1])
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            pid = results[-1][ind].get('pid', 'N/A')
            device = results[-1][ind].get('device', 'N/A')
            print(
                f"[{timestamp}] Process {i+1}/{num_process} | PID: {pid:<6} | GPU: {device:<3} | Results: {res_len}\n"
            )

        results = {k: v for d in results for k, v in d.items()}
        now_seed = np.array([v['seed'] for v in results.values()]).max()
        succ_num = succ_cnt.value

    with open(log_path, 'r', newline='') as f:
        lines = f.readlines()
    sumary = [
        f'Task Name: {task_name}\n',
        f"End time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}\n",
        f"Instruction Type: {instruction_type}\n",
        f'Success Rate: {succ_num / test_num} ({succ_num}/{test_num})\n',
        '\n'
    ]
    if len(lines) > 0:
        lines = sumary[0:1] + lines[0:1] + sumary[1:] + lines[1:]
    else: lines = sumary
    with open(log_path, 'w', newline='') as f:
        f.writelines(lines)
    print(f'Data has been saved to {log_path}')

    return now_seed, succ_num

def eval_policy_worker(
    task_name,
    args, model_args,
    seed, need, lock, succ_cnt, finish_cnt, 
    log_lock, log_path, return_queue,
    test_num=100,
    video_size=None,
    instruction_type=None,
    gpu_id=None,
):
    TASK_ENV = class_decorator(args["task_name"])
    policy_name = args["policy_name"]
    get_model = eval_function_decorator(policy_name, "get_model")
    eval_func = eval_function_decorator(policy_name, "eval")
    reset_func = eval_function_decorator(policy_name, "reset_model")
    model = get_model(model_args)

    expert_check = True
    TASK_ENV.suc = 0
    TASK_ENV.test_num = 0

    now_id = 0
    succ_seed = 0
    # suc_test_seed_list = []

    task_total_reward = 0
    clear_cache_freq = args["clear_cache_freq"]

    args["eval_mode"] = True
    log_results = {}
    render_freq = args["render_freq"]
    while need.value > 0:
        with lock:
            now_seed = seed.value
            seed.value += 1
        args["render_freq"] = 0
        if expert_check:
            try:
                TASK_ENV.setup_demo(now_ep_num=test_num-need.value, seed=now_seed, is_test=True, **args)
                episode_info = TASK_ENV.play_once()
                TASK_ENV.close_env()
            except UnStableError as e:
                TASK_ENV.close_env()
                args["render_freq"] = render_freq
                continue
            except Exception as e:
                TASK_ENV.close_env()
                args["render_freq"] = render_freq
                print("error occurs !")
                continue
        if (not expert_check) or (TASK_ENV.plan_success and TASK_ENV.check_success()):
            with lock:
                if need.value > 0:
                    now_id = test_num-need.value
                    need.value -= 1
                else: continue
            succ_seed += 1
            # suc_test_seed_list.append(now_seed)
            log_ind = now_id+1
            log_info = {'seed': now_seed, 'success': None}
            log_results[log_ind] = log_info
            args["render_freq"] = render_freq
            # get instaruction
            episode_info_list = [episode_info["info"]]
            results = generate_episode_descriptions(args["task_name"], episode_info_list, test_num)
            instruction = np.random.choice(results[0][instruction_type])

            eval_start = time.time()
            TASK_ENV.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)
            log_result(
                log_path, log_ind, log_info, log_lock, 
                start = eval_start, device = gpu_id, pid = os.getpid()
            )
            TASK_ENV.set_instruction(instruction=instruction)  # set language instruction
            if TASK_ENV.eval_video_path is not None:
                ffmpeg = subprocess.Popen(
                    [
                        "ffmpeg",
                        "-y",
                        "-loglevel",
                        "error",
                        "-f",
                        "rawvideo",
                        "-pixel_format",
                        "rgb24",
                        "-video_size",
                        video_size,
                        "-framerate",
                        "10",
                        "-i",
                        "-",
                        "-pix_fmt",
                        "yuv420p",
                        "-vcodec",
                        "libx264",
                        "-crf",
                        "23",
                        f"{TASK_ENV.eval_video_path}/episode{now_id}.mp4",
                    ],
                    stdin=subprocess.PIPE,
                )
                TASK_ENV._set_eval_video_ffmpeg(ffmpeg)
            success = False
            infer_cnt = 0
            limit = TASK_ENV.step_lim
            reset_func(model)
            while TASK_ENV.take_action_cnt < TASK_ENV.step_lim:
                infer_cnt += 1
                observation = TASK_ENV.get_obs()
                eval_func(TASK_ENV, model, observation)
                if TASK_ENV.eval_success:
                    success = True
                    succ_cnt.value += 1
                    break
            succ_num = succ_cnt.value
            finish_cnt.value += 1
            finish_num = finish_cnt.value
            # task_total_reward += TASK_ENV.episode_score
            if TASK_ENV.eval_video_path is not None:
                TASK_ENV._del_eval_video_ffmpeg()
            TASK_ENV.close_env(clear_cache=((succ_seed + 1) % clear_cache_freq == 0))
            if TASK_ENV.render_freq:
                TASK_ENV.viewer.close()
            TASK_ENV.test_num += 1

            eval_end = time.time()
            eval_time = eval_end - eval_start
            action_cnt = TASK_ENV.take_action_cnt
            if success:
                TASK_ENV.suc += 1
                print(f"\033[K\033[92mSuccess! {action_cnt} / {limit}\033[0m")
            else:
                print(f"\033[K\033[91mFail!    {action_cnt} / {limit}\033[0m")
            log_result(
                log_path, log_ind, log_info, log_lock, 
                success = success, end = eval_end, time = eval_time, fps = render_freq,
                action_cnt = action_cnt, limit = limit, action_gen_freq = action_cnt/eval_time,
                infer_cnt = infer_cnt, infer_freq = infer_cnt/eval_time
            )
        
            print(
                f"\033[K\033[93m{task_name}\033[0m | \033[94m{args['policy_name']}\033[0m | \033[92m{args['task_config']}\033[0m | \033[91m{args['ckpt_setting']}\033[0m\n"
                f"Success rate: \033[96m{succ_num}/{finish_num}\033[0m => \033[95m{round(succ_num/finish_num*100, 1)}%\033[0m, current seed: \033[90m{now_seed}\033[0m\n"
            )
    return_queue.put(log_results)
    return now_seed, TASK_ENV.suc

def parse_args_and_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--overrides", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Parse overrides
    def parse_override_pairs(pairs):
        override_dict = {}
        for i in range(0, len(pairs), 2):
            key = pairs[i].lstrip("--")
            value = pairs[i + 1]
            try:
                value = eval(value)
            except:
                pass
            override_dict[key] = value
        return override_dict

    if args.overrides:
        overrides = parse_override_pairs(args.overrides)
        config.update(overrides)

    return config


if __name__ == "__main__":
    from test_render import Sapien_TEST
    Sapien_TEST()

    usr_args = parse_args_and_config()

    main(usr_args)

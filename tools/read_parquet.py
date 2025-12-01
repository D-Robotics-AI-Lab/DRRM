import argparse
import json
import logging
import os

import pandas as pd
from tqdm import tqdm
import torch
from torchvision import transforms
from PIL import Image
import io

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vggt_model_path", "-p", type=str, 
                        default=os.environ.get('VGGT_CKP', None))
    parser.add_argument("--device", "-d", type=str, default="cpu")
    parser.add_argument("--steps", "-s", type=int, default=10)
    args = parser.parse_args()
    return args

def load_data(task_path, episode_path):
    transform = transforms.Compose([
        transforms.ToTensor()  # 自动转换PIL图像为Tensor，范围[0,1]并转为CHW格式
    ])
    # 2. 创建存储结果的列表
    batch_tensors = []
    # 优化选项（大文件适用）
    df = pd.read_parquet(
        os.path.join(task_path, episode_path),
        engine='pyarrow',       # 指定引擎（pyarrow/fastparquet）
        memory_map=True,        # 内存映射加速读取
        use_threads=True,       # 多线程并行
        # columns=['head_cam', 'front_cam']
    )
    print(df.head())
    print(df.info())
    with open(os.path.join(task_path, "meta/info.json")) as f:
        info = json.load(f)
    for _, row in df.iterrows():
        sample_tensors = []
        
        # 处理head_cam和front_cam
        came_keys = [k for k,v in info['features'].items() if v['dtype']=='image']
        # if 'thanos' in task_path: 
        #     came_keys.remove('head_cam')
        for cam_key in came_keys:
            # 从字节数据创建PIL图像
            img_data = row[cam_key]['bytes']
            img = Image.open(io.BytesIO(img_data))
            
            # 转换为RGB确保三通道
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # 应用转换并添加批次维度
            tensor_img = transform(img).unsqueeze(0)  # [1, 3, H, W]
            sample_tensors.append(tensor_img)
        
        # 沿序列维度(S)拼接两摄像头图像
        stacked_sample = torch.cat(sample_tensors, dim=0)  # [2, 3, H, W]
        batch_tensors.append(stacked_sample)

    # 4. 沿批次维度(B)合并所有样本
    final_tensor = torch.stack(batch_tensors, dim=0)  # [B, 2, 3, H, W]
    return final_tensor

def extract_img(task_path, episode_naem):
    transform = transforms.Compose([
        transforms.ToTensor()  # 自动转换PIL图像为Tensor，范围[0,1]并转为CHW格式
    ])
    # 2. 创建存储结果的列表
    episode_path = f"data/chunk-000/{episode_naem}.parquet"
    # 优化选项（大文件适用）
    df = pd.read_parquet(
        os.path.join(task_path, episode_path),
        engine='pyarrow',       # 指定引擎（pyarrow/fastparquet）
        memory_map=True,        # 内存映射加速读取
        use_threads=True,       # 多线程并行
        # columns=['head_cam', 'front_cam']
    )
    print(df.head())
    print(df.info())
    with open(os.path.join(task_path, "meta/info.json")) as f:
        info = json.load(f)
    for i, row in df.iterrows():
        sample_tensors = []
        
        # 处理head_cam和front_cam
        came_keys = [k for k,v in info['features'].items() if v['dtype']=='image']
        # if 'thanos' in task_path: 
        #     came_keys.remove('head_cam')
        for cam_key in came_keys:
            # 从字节数据创建PIL图像
            img_data = row[cam_key]['bytes']
            img = Image.open(io.BytesIO(img_data))
            
            # 转换为RGB确保三通道
            if img.mode != 'RGB':
                img = img.convert('RGB')

            # resize img to 427 * 240, cut width center 320
            img = img.resize((427, 240))
            left = (427 - 320) / 2
            right = left + 320
            img = img.crop((left, 0, right, 240))

            # 将img存储至本地文件
            os.makedirs(f'{task_path}/imgs/{episode_naem}', exist_ok=True)
            img.save(f'{task_path}/imgs/{episode_naem}/{i:06d}_{cam_key}.png')
            



def inference_log(verbo = True):
    if verbo:
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
    args = get_args()
    
    # dataset, task, episode = 'lerobot', 'block_hammer_beat_D435_200', 'episode_000166'
    # dataset, task, episode = 'lerobot', 'bottle_adjust_D435_200', 'episode_000188'
    # dataset, task, episode = 'lerobot', 'container_place_D435_200', 'episode_000004'
    # dataset, task, episode = 'lerobot', 'put_apple_cabinet_D435_200', 'episode_000143'
    # dataset, task, episode = 'thanos', 'stack_cubes', 'episode_000098'
    # dataset, task, episode = 'thanos', 'obs_test', 'episode_000001'
    # dataset, task, episode = 'thanos', 'place_cube_limit_hr', 'episode_000001'
    dataset, task, episode = 'thanos', 'place_tiny_cube_200', 'episode_000001'
    
    vggt_target = 'Pi3'
    vggt_target = 'VGGT'
    limit = 1
    emvis_config = {
        'dim_2d': None, # image embedding dimension, default 1024
        'load_vggt_pretrain': True,
        'load_vggt_heads': True,
        'vggt_target': vggt_target,
        # 'vggt_model_path': args.vggt_model_path,
        'mlp_ratio': 1.0, # expend ration for hidden layer of token adapter
        'drop_p': 0.,
        'seq_as_view': False,
        'intermediate_layer_idx': [23], # list of layer index of vggt encoder output
        'only_first_view': True,
        'only_2d': True,
        'interpolate': 'bilinear',
        'injector_config': {},
        'model_adapter_config': {},
        'mv_fuser_config': {},
        'visualize': True,
        'return_predict': False
    }
    # third party encoder config
    if vggt_target == 'VGGT': 
        emvis_config['vggt_model_path'] = args.vggt_model_path
        emvis_config['intermediate_layer_idx'] = [4, 11, 17, 23]
    scene_encoder = EmVisRM(**emvis_config).to(device=args.device)
    scene_encoder.eval()

    for task in os.listdir(f"datasets/{dataset}/"):
        if '3d' in task or task in ['obs_test', 'cache']: continue
        for i, episode in enumerate(os.listdir(os.path.join(f"datasets/{dataset}/{task}", f"data/chunk-000"))):
            if i+1 > limit: break
            episode = episode.split('.')[0]
            data = load_data(f"datasets/{dataset}/{task}", f"data/chunk-000/{episode}.parquet").to(device=args.device)
            for i in tqdm(range(len(data))):
                os.environ["DEBUG_DIR"] = f'out/{vggt_target}/{task}/{episode}/bi-view'
                batch = data[i:i+1]
                _ = scene_encoder(batch)
                
                os.environ["DEBUG_DIR"] = f'out/{vggt_target}/{task}/{episode}/single-view'
                batch = data[i:i+1,0:1]
                _ = scene_encoder(batch)

if __name__ == "__main__":
    data = extract_img(f"datasets/thanos/stack_cubes_3d_200", f"episode_000000")
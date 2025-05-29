from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from drrm.common.normalizer import LinearNormalizer
from drrm.common.normalize_util import get_image_range_normalizer, get_range_normalizer_from_stat, get_identity_normalizer_from_stat
from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
from pathlib import Path
import torch
import numpy as np
import shutil
from lerobot.common.datasets.compute_stats import sample_images, get_feature_stats
from lerobot.common.datasets.utils import (
    validate_episode_buffer,
    get_episode_data_index,
    check_timestamps_sync,
)

def downsample_mask(mask, max_n, seed=0):
    # subsample training data
    train_mask = mask
    if (max_n is not None) and (np.sum(train_mask) > max_n):
        n_train = int(max_n)
        curr_train_idxs = np.nonzero(train_mask)[0]
        rng = np.random.default_rng(seed=seed)
        train_idxs_idx = rng.choice(len(curr_train_idxs), size=n_train, replace=False)
        train_idxs = curr_train_idxs[train_idxs_idx]
        train_mask = np.zeros_like(train_mask)
        train_mask[train_idxs] = True
        assert np.sum(train_mask) == n_train
    return train_mask

def get_val_mask(n_episodes, val_ratio, seed=0):
    val_mask = np.zeros(n_episodes, dtype=bool)
    if val_ratio <= 0:
        return val_mask

    # have at least 1 episode for validation, and at least 1 episode for train
    n_val = min(max(1, round(n_episodes * val_ratio)), n_episodes-1)
    rng = np.random.default_rng(seed=seed)
    val_idxs = rng.choice(n_episodes, size=n_val, replace=False)
    val_mask[val_idxs] = True
    return val_mask

def decode_string_to_tensor(encoded_str: str) -> torch.Tensor:
    """使用pickle从string解码tensor，更高效的方法"""
    import pickle
    import base64
    
    # 从base64解码
    pickled_bytes = base64.b64decode(encoded_str)
    # 用pickle反序列化
    numpy_array = pickle.loads(pickled_bytes)
    # 转换为torch tensor
    return torch.from_numpy(numpy_array)

def compute_episode_stats(episode_data: dict[str, list[str] | np.ndarray], features: dict) -> dict:
    ep_stats = {}
    for key, data in episode_data.items():
        if features[key]["dtype"] == "string":
            continue  # HACK: we should receive np.arrays of strings
        elif features[key]["dtype"] in ["image", "video"]:
            ep_ft_array = sample_images(data)  # data is a list of image paths
            axes_to_reduce = (0, 2, 3)  # keep channel dim
            keepdims = True
        else:
            if len(features[key]['shape']) >= 2:
                ep_ft_array = data  # data is already a np.ndarray
                axes_to_reduce = tuple(range(len(data.shape) - 1)) # compute stats over the first axis
                keepdims = True 
            else:
                ep_ft_array = data  # data is already a np.ndarray
                axes_to_reduce = 0  # compute stats over the first axis
                keepdims = data.ndim == 1  # keep as np.array

        ep_stats[key] = get_feature_stats(ep_ft_array, axis=axes_to_reduce, keepdims=keepdims)

        # finally, we normalize and remove batch dim for images
        if features[key]["dtype"] in ["image", "video"]:
            ep_stats[key] = {
                k: v if k == "count" else np.squeeze(v / 255.0, axis=0) for k, v in ep_stats[key].items()
            }

    return ep_stats

class DRRMDataset(LeRobotDataset):
    def __init__(self, 
        repo_id: str,
        root: str | Path | None = None,
        dataset_metadata: dict | None = None,
        episodes: list[int] | None = None,
        horizon: int = 8,
        pad_before: int = 0,
        pad_after: int = 0,
        npy_feature_keys: list[str] | None = None,
        seed: int = 0,
        val_ratio: float = 0.0,
        max_train_episodes: int | None = None,
        use_vggt_input: bool = False,
        ):
        # 只调用LeRobotDataset的初始化，因为BaseDataset的__init__是空的
        self.dataset_meta = dataset_metadata
        self.repo_id = repo_id
        # 保存构造参数以便创建验证集
        self.root = root
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.seed = seed
        self.val_ratio = val_ratio
        self.max_train_episodes = max_train_episodes
        self.npy_feature_keys = npy_feature_keys if npy_feature_keys is not None else []
        self.use_vggt_input = use_vggt_input
        self.default_keys = {'timestamp', 'frame_indx', 'episode_index', 'index', 'task_index', 'frame_index'}
        # 计算各种key对应的delta_timestamps
        self.delta_timestamps = {
            key: [t / self.dataset_meta.fps for t in range(-pad_before, -pad_before+horizon)]
            for key in self.dataset_meta.names.keys() if key not in self.default_keys
        }
        self.val_mask = None
        if episodes is None:
            n_episodes = self.dataset_meta.total_episodes
            self.val_mask = get_val_mask(n_episodes, val_ratio, seed)
            train_mask = ~self.val_mask
            train_mask = downsample_mask(train_mask, max_train_episodes, seed)
            episodes = np.nonzero(train_mask)[0]
        self.ep_idx_to_arr_idx = {ep_idx: arr_idx for arr_idx, ep_idx in enumerate(episodes)}
        LeRobotDataset.__init__(self, 
                repo_id = repo_id, 
                root = root, 
                episodes = episodes,
                delta_timestamps = self.delta_timestamps)

    def get_validation_dataset(self):
        if self.val_mask is None:
            return None
        
        # 获取验证集的episode索引
        val_episodes = np.nonzero(self.val_mask)[0]
        
        # 创建验证集dataset实例
        val_dataset = self.__class__(
            repo_id=self.repo_id,
            root=self.root,
            dataset_metadata=self.dataset_meta,
            episodes=val_episodes.tolist(),
            horizon=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            npy_feature_keys=self.npy_feature_keys,
            seed=self.seed,
            val_ratio=0.0,  # 验证集不需要再分割
            max_train_episodes=None,
            use_vggt_input=self.use_vggt_input
        )
        return val_dataset

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        # 组合成各种normalizer
        normalizer = LinearNormalizer()
        for key in self.npy_feature_keys:
            stat = {
                    'min': np.float32([-1e8]),
                    'max': np.float32([1e8]),
                    'mean': np.float32([0]),
                    'std': np.float32([1e8]),
            }
            normalizer[key] = get_identity_normalizer_from_stat(stat)

        
        for key in self.dataset_meta.names.keys():
            if key not in self.default_keys:
                if self.dataset_meta.features[key]['dtype'] in ['image', 'video']:
                    normalizer[key] = get_image_range_normalizer()
                elif self.dataset_meta.features[key]['dtype'] == 'string':
                    stat = {
                            'min': np.float32([-1e8]),
                            'max': np.float32([1e8]),
                            'mean': np.float32([0]),
                            'std': np.float32([1e8]),
                        }
                    normalizer[key] = get_identity_normalizer_from_stat(stat)
                else:
                    stat = {
                        'min': self.dataset_meta.stats[key]['min'].astype(np.float32),
                        'max': self.dataset_meta.stats[key]['max'].astype(np.float32),
                        'mean': self.dataset_meta.stats[key]['mean'].astype(np.float32),
                        'std': self.dataset_meta.stats[key]['std'].astype(np.float32),
                    }
                    normalizer[key] = get_range_normalizer_from_stat(stat)
        return normalizer
    
    def _query_npy_frame(self, query_indices: dict[str, list[int]], ep_idx: int) -> dict:
        relative_ep_idx = self.ep_idx_to_arr_idx[ep_idx]
        indices = [idx - self.episode_data_index["from"][relative_ep_idx].item() for idx in query_indices['action']]
        lerobot_root = self.dataset_meta.root
        result = {}
        for key in self.npy_feature_keys:
            file_path = lerobot_root / f"npy/{key}/{ep_idx}.npy"
            arr = np.load(file_path, mmap_mode='r')
            result[key] = arr[indices].copy()
        return result

    # def _query_hf_dataset(self, query_indices: dict[str, list[int]]) -> dict:
    #     result = {}
    #     import time

        
    #     for key, q_idx in query_indices.items():
    #         if self.dataset_meta.features[key]['dtype'] not in ['image', 'video']:
    #             if len(self.dataset_meta.features[key]['shape']) >= 2:
    #                 result[key] = torch.stack(self.hf_dataset.select(q_idx)[key]).squeeze(-1)
    #             else:
    #                 result[key] = torch.stack(self.hf_dataset.select(q_idx)[key])

    #     return result

    def save_episode(self, episode_data: dict | None = None) -> None:
        """
        This will save to disk the current episode in self.episode_buffer.

        Args:
            episode_data (dict | None, optional): Dict containing the episode data to save. If None, this will
                save the current episode in self.episode_buffer, which is filled with 'add_frame'. Defaults to
                None.
        """
        if not episode_data:
            episode_buffer = self.episode_buffer

        validate_episode_buffer(episode_buffer, self.meta.total_episodes, self.features)

        # size and task are special cases that won't be added to hf_dataset
        episode_length = episode_buffer.pop("size")
        tasks = episode_buffer.pop("task")
        episode_tasks = list(set(tasks))
        episode_index = episode_buffer["episode_index"]

        episode_buffer["index"] = np.arange(self.meta.total_frames, self.meta.total_frames + episode_length)
        episode_buffer["episode_index"] = np.full((episode_length,), episode_index)

        # Add new tasks to the tasks dictionary
        for task in episode_tasks:
            task_index = self.meta.get_task_index(task)
            if task_index is None:
                self.meta.add_task(task)

        # Given tasks in natural language, find their corresponding task indices
        episode_buffer["task_index"] = np.array([self.meta.get_task_index(task) for task in tasks])

        for key, ft in self.features.items():
            # index, episode_index, task_index are already processed above, and image and video
            # are processed separately by storing image path and frame info as meta data
            if key in ["index", "episode_index", "task_index"] or ft["dtype"] in ["image", "video"]:
                continue
            episode_buffer[key] = np.stack(episode_buffer[key])

        self._wait_image_writer()
        self._save_episode_table(episode_buffer, episode_index)
        ep_stats = compute_episode_stats(episode_buffer, self.features)

        if len(self.meta.video_keys) > 0:
            video_paths = self.encode_episode_videos(episode_index)
            for key in self.meta.video_keys:
                episode_buffer[key] = video_paths[key]

        # `meta.save_episode` be executed after encoding the videos
        self.meta.save_episode(episode_index, episode_length, episode_tasks, ep_stats)

        ep_data_index = get_episode_data_index(self.meta.episodes, [episode_index])
        ep_data_index_np = {k: t.numpy() for k, t in ep_data_index.items()}
        check_timestamps_sync(
            episode_buffer["timestamp"],
            episode_buffer["episode_index"],
            ep_data_index_np,
            self.fps,
            self.tolerance_s,
        )

        video_files = list(self.root.rglob("*.mp4"))
        assert len(video_files) == self.num_episodes * len(self.meta.video_keys)

        parquet_files = list(self.root.rglob("*.parquet"))
        assert len(parquet_files) == self.num_episodes

        # delete images
        img_dir = self.root / "images"
        if img_dir.is_dir():
            shutil.rmtree(self.root / "images")

        if not episode_data:  # Reset the buffer
            self.episode_buffer = self.create_episode_buffer()

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """
        重写__getitem__方法,组合两个父类的功能
        """
        # 首先调用LeRobotDataset的__getitem__获取原始数据
        # import time 
        # start_time = time.time()
        item = self.hf_dataset[idx]
        ep_idx = item["episode_index"].item()
        # current_time = time.time()
        # print(f"Query HF dataset elapsed time: {current_time - start_time:.4f} seconds")
        query_indices = None
        if self.delta_indices is not None:
            query_indices, padding = self._get_query_indices(idx, self.ep_idx_to_arr_idx[ep_idx])
            query_result = self._query_hf_dataset(query_indices)
            item = {**item, **padding}
            for key, val in query_result.items():
                item[key] = val

        npy_frame = self._query_npy_frame(query_indices, ep_idx)
        item = {**npy_frame, **item}
        if len(self.meta.video_keys) > 0:
            current_ts = item["timestamp"].item()
            query_timestamps = self._get_query_timestamps(current_ts, query_indices)
            video_frames = self._query_videos(query_timestamps, ep_idx)
            item = {**video_frames, **item}

        if self.image_transforms is not None:
            image_keys = self.meta.camera_keys
            for cam in image_keys:
                item[cam] = self.image_transforms(item[cam])

        # Add task as a string
        task_idx = item["task_index"].item()
        item["task"] = self.meta.tasks[task_idx]
        # 然后按照BaseDataset的期望格式重新组织数据
        data = {
            'obs': {
                'agent_pos': item['agent_pos'],
            },
            'action': item['action'],
        }
        if self.use_vggt_input:
            data['obs']['head_cam'] = item['head_cam'] 
        else:
            data['obs']['point_cloud'] = item['point_cloud']

        for key in self.npy_feature_keys:
            if key in item:
                data['obs'][key] = item[key]
        # print(f"Query data elapsed time: {time.time() - start_time:.4f} seconds")
        return data

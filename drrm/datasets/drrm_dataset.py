from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
from pathlib import Path
import numpy as np
import shutil
from lerobot.common.datasets.compute_stats import sample_images, get_feature_stats
from lerobot.common.datasets.utils import (
    validate_episode_buffer,
    get_episode_data_index,
    check_timestamps_sync,
)
from drrm.common.normalizer import LinearNormalizer
from drrm.common.normalize_util import (
    get_image_range_normalizer, 
    get_range_normalizer_from_stat, 
    get_identity_normalizer_from_stat
)

def downsample_mask(mask, max_n, seed=0):
    """Downsample training data to max_n samples"""
    if max_n is None or np.sum(mask) <= max_n:
        return mask
    
    n_train = int(max_n)
    curr_train_idxs = np.nonzero(mask)[0]
    rng = np.random.default_rng(seed=seed)
    train_idxs_idx = rng.choice(len(curr_train_idxs), size=n_train, replace=False)
    train_idxs = curr_train_idxs[train_idxs_idx]
    
    new_mask = np.zeros_like(mask)
    new_mask[train_idxs] = True
    return new_mask


def get_val_mask(n_episodes, val_ratio, seed=0):
    """Create validation mask for episodes"""
    if val_ratio <= 0:
        return np.zeros(n_episodes, dtype=bool)

    # Ensure at least 1 episode for validation and 1 for training
    n_val = min(max(1, round(n_episodes * val_ratio)), n_episodes - 1)
    rng = np.random.default_rng(seed=seed)
    val_idxs = rng.choice(n_episodes, size=n_val, replace=False)
    
    val_mask = np.zeros(n_episodes, dtype=bool)
    val_mask[val_idxs] = True
    return val_mask


def compute_episode_stats(episode_data, features):
    """Compute statistics for episode data"""
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
    """Enhanced LeRobot dataset with DRRM-specific features"""
    
    DEFAULT_KEYS = {'timestamp', 'frame_indx', 'episode_index', 'index', 'task_index', 'frame_index'}
    
    def __init__(self, 
            repo_id: str,
            root:  str | Path | None = None,
            dataset_metadata: LeRobotDatasetMetadata | None = None,
            episodes: list[int] | None = None,
            task_list: list[int] | None = None,
            task_index_list: list[int] | None = None,
            black_index: list[int] | None = None,
            horizon: int = 8,
            pad_before: int = 0,
            pad_after: int = 0,
            npy_feature_keys: list[str] | None = None,
            seed: int = 0,
            val_ratio: float = 0.0,
            max_train_episodes=None,
            **kwargs
        ):
        
        # Store all parameters for easy access
        self.dataset_meta = dataset_metadata
        self.repo_id = repo_id
        self.root = root
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.seed = seed
        self.val_ratio = val_ratio
        self.max_train_episodes = max_train_episodes
        self.npy_feature_keys = npy_feature_keys or []
        self.val_mask = None
        self.task_list = task_list
        self.task_index_list = task_index_list
        self.black_index = black_index
        
        # Setup delta timestamps
        self.delta_timestamps = self._create_delta_timestamps()
        
        # Handle episode selection and train/val split
        if episodes is None:
            episodes = self._setup_train_val_split()
            
        self.ep_idx_to_arr_idx = {ep_idx: arr_idx for arr_idx, ep_idx in enumerate(episodes)}
        
        # Initialize parent class
        super().__init__(
            repo_id=repo_id, 
            root=root, 
            episodes=episodes,
            delta_timestamps=self.delta_timestamps
        )

    def _create_delta_timestamps(self):
        """Create delta timestamps for all non-default keys"""
        timestamps = {}
        for key in self.dataset_meta.names.keys():
            if key not in self.DEFAULT_KEYS:
                timestamps[key] = [
                    t / self.dataset_meta.fps 
                    for t in range(-self.pad_before, -self.pad_before + self.horizon)
                ]
        return timestamps

    def _setup_train_val_split(self):
        """Setup train/validation split and return train episodes"""
        tn_episodes = self.dataset_meta.total_episodes
        available_mask = np.zeros(tn_episodes, dtype=bool)
        self.val_mask = np.zeros(tn_episodes, dtype=bool)
        self.train_mask = np.zeros(tn_episodes, dtype=bool)
        # task list -> task index list
        if self.task_index_list is not None:
            if self.task_list is None:
                self.task_list = []
            self.task_list += [self.dataset_meta.tasks[ind] for ind in self.task_index_list]
        if self.task_list is None:
            self.task_list = list(self.dataset_meta.tasks.values())
        # task index list -> available_mask
        available_mask[[
            bool(set(ep['tasks']) & set(self.task_list))
            for ep in self.dataset_meta.episodes.values()
        ]] = True
        if self.black_index is not None:
            available_mask[self.black_index] = False

        
        n_episodes = available_mask.sum()
        val_mask = get_val_mask(n_episodes, self.val_ratio, self.seed)
        self.val_mask[available_mask] = val_mask
        train_mask = ~val_mask
        train_mask = downsample_mask(train_mask, self.max_train_episodes, self.seed)
        self.train_mask[available_mask] = train_mask
        return np.nonzero(self.train_mask)[0]

    def get_validation_dataset(self):
        """Create validation dataset if validation split exists"""
        if self.val_mask is None:
            return None
        
        val_episodes = np.nonzero(self.val_mask)[0]
        
        return self.__class__(
            repo_id=self.repo_id,
            root=self.root,
            dataset_metadata=self.dataset_meta,
            episodes=val_episodes.tolist(),
            horizon=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            npy_feature_keys=self.npy_feature_keys,
            seed=self.seed,
            val_ratio=0.0,  # No further splitting for validation
            max_train_episodes=None,
        )

    def get_normalizer(self, **kwargs):
        """Create normalizer for all dataset features"""
        normalizer = LinearNormalizer()
        
        # Add VGGT feature normalizers (identity)
        for key in self.npy_feature_keys:
            normalizer[key] = self._create_identity_normalizer()
        
        # Add dataset feature normalizers
        for key in self.dataset_meta.names.keys():
            if key in self.DEFAULT_KEYS:
                continue
                
            feature = self.dataset_meta.features[key]
            normalizer[key] = self._create_feature_normalizer(key, feature)
                
        return normalizer

    def _create_identity_normalizer(self):
        """Create identity normalizer with wide bounds"""
        stat = {
            'min': np.float32([-1e8]),
            'max': np.float32([1e8]),
            'mean': np.float32([0]),
            'std': np.float32([1e8]),
        }
        return get_identity_normalizer_from_stat(stat)

    def _create_feature_normalizer(self, key, feature):
        """Create appropriate normalizer based on feature type"""
        if feature['dtype'] in ['image', 'video']:
            return get_image_range_normalizer()
        elif feature['dtype'] == 'string':
            return self._create_identity_normalizer()
        else:
            stats = self.dataset_meta.stats[key]
            stat = {
                'min': stats['min'].astype(np.float32),
                'max': stats['max'].astype(np.float32),
                'mean': stats['mean'].astype(np.float32),
                'std': stats['std'].astype(np.float32),
            }
            return get_range_normalizer_from_stat(stat)
    
    def _query_npy_frame(self, query_indices, ep_idx):
        """Load npy features from numpy files"""
        if not self.npy_feature_keys or query_indices is None:
            return {}
            
        relative_ep_idx = self.ep_idx_to_arr_idx[ep_idx]
        indices = [
            idx - self.episode_data_index["from"][relative_ep_idx].item() 
            for idx in query_indices['action']
        ]
        
        result = {}
        lerobot_root = self.dataset_meta.root
        
        for key in self.npy_feature_keys:
            file_path = lerobot_root / f"npy/{key}/{ep_idx}.npy"
            if not file_path.exists():
                file_path = lerobot_root / f"npy/{key}/episode_{ep_idx:06d}.npy"
            # 使用mmeap_mode + take
            with open(file_path, 'rb') as f:
                version = np.lib.format.read_magic(f)
                shape, fortran, dtype = np.lib.format._read_array_header(f, version)
                offset = f.tell()
            arr_memmap = np.memmap(file_path, dtype=dtype, mode='r', offset=offset).reshape(shape)
            result[key] = np.take(arr_memmap, indices, axis=0)
            
        return result

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

    def __getitem__(self, idx):
        """Get dataset item with all features"""
        # Get base item from HuggingFace dataset
        item = self.hf_dataset[idx] # idx是新数据集上的index
        ep_idx = item["episode_index"].item() # 根据item["episode_index"]找到旧数据集上的episode_index
        
        # Query temporal data if needed
        query_indices = None
        if self.delta_indices is not None:
            query_indices, padding = self._get_query_indices(idx, self.ep_idx_to_arr_idx[ep_idx])
            query_result = self._query_hf_dataset(query_indices)
            item.update(padding)
            item.update(query_result)

        # Add npy features from numpy files
        npy_frame = self._query_npy_frame(query_indices, ep_idx)
        item.update(npy_frame)
        
        # Add video frames if needed
        if len(self.meta.video_keys) > 0:
            current_ts = item["timestamp"].item()
            query_timestamps = self._get_query_timestamps(current_ts, query_indices)
            video_frames = self._query_videos(query_timestamps, ep_idx)
            item.update(video_frames)

        # Apply image transforms
        if self.image_transforms is not None:
            for cam in self.meta.camera_keys:
                item[cam] = self.image_transforms(item[cam])

        # Add task string
        task_idx = item["task_index"].item()
        item["task"] = self.meta.tasks[task_idx]
        
        # Organize into expected output format
        return self._organize_item_data(item)

    def _organize_item_data(self, item):
        """Organize raw item data into expected format"""
        data = {
            'obs': {},
            'action': item['action'],
        }

        # Add features to observations
        for key in item:
            if key in {*self.npy_feature_keys, *self.dataset_meta.names.keys()} and key not in {*self.DEFAULT_KEYS, 'action'}:
                data['obs'][key] = item[key]
                
        return data

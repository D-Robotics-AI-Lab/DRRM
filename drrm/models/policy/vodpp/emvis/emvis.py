from functools import partial
import logging
import time
from typing import Callable, Dict, List
import numpy as np
from torch import Tensor, nn
import torch
import os
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

from .utils import format_number, preprocess_images, param_count, get_processed_shape, save_depth, save_images, process_and_save
from .injector import DimInjector
from .layers import MLP
from .adapter import Adapter
from .mv_fuser import MultiViewFuser
# ModelAdapter = partial(Adapter, __target__='CNNAdapter')

logger = logging.getLogger(__name__)

def check_model_adapter_config(config):
    # {
    #     'image_shape': image_shape, # H, W
    #     'tile_shape': patch_shape, # image patch shape
    #     'tile_dim': patch_shape, # image patch shape
    #     'target_shape': target_shape, # (1)
    #     'target_dim': target_dim,
    #     'use_pixelshuffle': False,
    # }
    requires = ['dim_in', 'shape_out', 'dim_out']
    for key in requires:
        assert key in config, \
            f"Missing `{key}` in model adapter config"
    
    # image_shape = config.pop('image_shape', None)
    # # if config.get('tile_shape', None) == None:
    # #     assert image_shape != None, \
    # #         f"Missing `image_shape` or `tile_shape` in model adapter config"
    # #     resize_shape = get_processed_shape(*image_shape)
    # #     config['tile_shape'] = (resize_shape[0]//14, resize_shape[1]//14)
    # for key1, key2 in [
    #     ('tile_shape',None), 
    #     ('tile_dim','dim_in'), 
    #     ('target_shape','shape_out'), 
    #     ('target_dim','dim_out')
    # ]:
    #     if key2:
    #         config[key2] = config.pop(key1)
    #     else:
    #         config.pop(key1)


class EmVisRM(nn.Module):
    def __init__(
        self, 
        dim_2d: int = 1024, # image embedding dimension
        dim_3d_keys: list = ['frame', 'global'], # image embedding dimension
        vggt_target: str = 'VGGT',
        load_vggt_pretrain = False,
        vggt_model_path = None,
        mlp_ratio = 4.0,
        drop_p = 0.,
        ffn_layer_num = 1,
        # preprocessing config
        processor_config: Dict = {},
        interpolate: str = 'bilinear', # None nearest bilinear(default) bicubic
        # feature config
        only_2d: bool = False,
        seq_as_view: bool = False,
        view_as_seq: bool = False,
        return_predict: bool = False,
        only_first_view: bool = False,
        # vggt config
        intermediate_layer_idx: List = [4, 11, 17, 23],
        ft_layer_idx: List = [],
        vggt_heads_list: List = ['camera_head', 'point_head', 'depth_head', 'track_head'],
        ft_heads: List = [],
        vggt_config: Dict = {},
        # module config
        injector_config: Dict = {},
        model_adapter_config: Dict = None,
        mv_fuser_config: Dict = None,
        visualize: bool = False,
        **kwargs
    ):
        super().__init__()
        self.img_processing = partial(preprocess_images, interpolate=interpolate)
        self.dim_2d = dim_2d if dim_2d!=None else 1024
        assert len(dim_3d_keys) <= 2
        self.dim_3d = len(dim_3d_keys) * 1024
        self.view_as_seq = view_as_seq

        # initialize VGGT
        self.seq_as_view = seq_as_view
        self.vggt_target = vggt_target.lower()
        if self.vggt_target == 'pi3':
            from .pi3_encoder import Pi3Encoder, Pi3Heads
            self.vggt_encoder = Pi3Encoder(
                ft_layer_idx = ft_layer_idx,
                intermediate_layer_idx=intermediate_layer_idx,
                dim_keys = dim_3d_keys,
            )
            self.vggt_heads = self.vggt_encoder.get_pi3_heads(
                ft_heads = ft_heads,
                heads=vggt_heads_list
            )
            self.img_processing = partial(preprocess_images, interpolate=interpolate)
        elif self.vggt_target == 'da3':
            from .da3_encoder import DA3Encoder
            # 读 vggt_model_path 下面的config.json 获取 model_name
            import json
            with open(os.path.join(vggt_model_path, "config.json"), "r") as f:
                config = json.load(f)
                model_name = config.get("model_name", vggt_model_path)
            self.vggt_encoder = DA3Encoder(
                ft_layer_idx = ft_layer_idx,
                intermediate_layer_idx=intermediate_layer_idx,
                dim_keys = dim_3d_keys,
                model_name = model_name
            )
            if hasattr(self.vggt_encoder.model, 'da3'):
                self.vggt_encoder.model.da3.head = None
                self.vggt_encoder.model.da3.cam_enc = None
                self.vggt_encoder.model.da3.cam_dec = None
                self.vggt_encoder.model.da3.gs_head = None
                self.vggt_encoder.model.da3_metric = None
                self.vggt_heads = None
            else:
                self.vggt_encoder.model.head = None
                self.vggt_encoder.model.cam_enc = None
                self.vggt_encoder.model.cam_dec = None
                self.vggt_encoder.model.gs_head = None
                self.vggt_encoder.model.gs_adapter = None
                self.vggt_heads = None
                
            if isinstance(interpolate, int):
                self.img_processing = partial(self.vggt_encoder.da3_preprocess, process_res=interpolate)
            else:
                self.img_processing = self.vggt_encoder.da3_preprocess
        elif 'dinov3' in self.vggt_target:
            if 'cnn' in self.vggt_target:
                from .dinov3cnn_encoder import DINOv3Encoder
                from transformers import DINOv3ConvNextConfig as DINOv3Config
            else:
                from .dinov3_encoder import DINOv3Encoder
                from transformers import DINOv3ViTConfig as DINOv3Config
            from transformers import DINOv3ViTImageProcessorFast
            from transformers import AutoImageProcessor, AutoModel, AutoConfig
            if vggt_config:
                config = DINOv3Config(**vggt_config)
            else:
                config = AutoConfig.from_pretrained(f"{vggt_model_path}")
                vggt_config.update(config.to_diff_dict())

            if processor_config:
                processor =  DINOv3ViTImageProcessorFast(**processor_config)
            else:
                processor = AutoImageProcessor.from_pretrained(f"{vggt_model_path}")
                processor_config.update(processor.to_dict())
            self.vggt_encoder = DINOv3Encoder(config)
            self.vggt_heads = None
            self.img_processing = lambda x: processor(x, return_tensors="pt")['pixel_values']
        else:
            from .vggt_encoder import VGGTEncoder
            from .vggt_heads import VGGTHead
            self.vggt_encoder = VGGTEncoder(
                ft_layer_idx = ft_layer_idx,
                intermediate_layer_idx=intermediate_layer_idx,
                dim_keys = dim_3d_keys,
            )
            self.vggt_heads = VGGTHead(
                ft_heads = ft_heads,
                heads=vggt_heads_list
            )
            self.img_processing = partial(preprocess_images, interpolate=interpolate)
        if load_vggt_pretrain:
            self.vggt_encoder.load_pretrained_model(vggt_model_path)
            if self.vggt_heads != None: 
                self.vggt_heads.load_pretrained_model(vggt_model_path)

        self.visualize = visualize
        self.intermediate_layer_idx = intermediate_layer_idx
        # initialize injector
        self.only_first_view = only_first_view
        self.only_2d = only_2d
        if self.only_2d:
            self.injector = None
            dim_now = self.dim_2d
        elif injector_config:
            injector_config = {
                'dim_3d': self.dim_3d,
                'dim_2d': self.dim_2d,
                # 'dim_out': self.dim_3d (default)
                'rope': self.vggt_encoder.rope,
                'mlp_ratio': mlp_ratio,
                'drop_p': drop_p,
                'ffn_layer_num': ffn_layer_num,
                'intermediate_layer_idx': intermediate_layer_idx,
                ** injector_config
            }
            self.injector = DimInjector(**injector_config)
            dim_now = injector_config['dim_out'] if 'dim_out' in injector_config \
                else self.dim_3d
        else:
            self.injector = None
            # assert len(self.intermediate_layer_idx) == 1, "Only could use one layer feature"
            dim_now = 128

        # initialize multiview fuser
        if mv_fuser_config:
            mv_fuser_config = {
                'dim_in': dim_now,
                'mlp_ratio': mlp_ratio,
                'drop_p': drop_p,
                'ffn_layer_num': ffn_layer_num,
                ** mv_fuser_config
            }
            self.mv_fuser = MultiViewFuser(
                **mv_fuser_config
            )
            dim_now = mv_fuser_config['dim_out'] if 'dim_out' in injector_config \
                else dim_now
        else: self.mv_fuser = None
        
        # initialize model adapter
        if model_adapter_config:
            model_adapter_config = {
                'dim_in': dim_now,
                **model_adapter_config
            }
            check_model_adapter_config(model_adapter_config)
            self.model_adapter = Adapter(
                **model_adapter_config
            )
            dim_now = model_adapter_config['dim_out']
        else: self.model_adapter = None
        self.dim_out = dim_now
        self.shape_out = model_adapter_config['shape_out']

        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        trainable_ratio = (trainable_params / total_params) * 100 if total_params > 0 else 0.0
        params_state = '\n'.join(['  '+t for t in param_count(self).split('\n')])
        log_msg = (
            f"EmVisRM Parameters:\n"
            f"  Total: {format_number(total_params)} ({total_params:,})\n"
            f"  Trainable: {format_number(trainable_params)} ({trainable_params:,})\n"
            f"  Frozen: {format_number(frozen_params)} ({frozen_params:,})\n"
            f"  Trainable Ratio: {trainable_ratio:.2f}%\n"
            f"{params_state}"
        )
        logger.info(log_msg)
        self.return_predict = return_predict
    
    def image_forward(self, origin_images: Tensor, image_tokens: Tensor = None, image_pos: Tensor = None):
        # images: [B * S, V, 3, H, W] RGB tensor
        # image_tokens: [B * S, V, P_image, D]
        # image_pos: [B * S, V, P_image, 2]
        time0 = time.time()
        BS, V, _, H_origin, W_origin = origin_images.size()
        images = self.img_processing(origin_images.view(-1, 3, H_origin, W_origin))
        _, _, H, W = images.size()
        images = images.view(BS, V, 3, H, W)
        
        if image_tokens != None:
            BS, V, P_i, D = image_tokens.shape
            assert D == self.dim_2d, \
                "Dimension mismatch: When passing custom image tokens, you must specify dim_2d in initialization."
            assert image_pos.shape == (BS, V, P_i, 2), \
                f"The image patch position tensor should be the shape with [{BS}, {V}, {P_i}, {2}]"
        else:
            logger.debug("The input `image_tokens` is None, using the built-in image encoder.")
        time1 = time.time()
        logger.debug(f"Image preprocess time: {time1-time0:.5f} seconds")

        ### images -> VGGT ENCODER -> vggt_token_dict
        vggt_token_dict = self.vggt_encoder(images, V_fake=self.view_fake)
        time2 = time.time()
        logger.debug(f"VGGT encoding time: {time2-time1:.5f} seconds")
        if self.vggt_heads and not self.training and self.visualize:
            self.visualization(images, vggt_token_dict, True)
        # self.visualization(images, vggt_token_dict)
        return self.vggt_forward(vggt_token_dict, image_tokens, image_pos)
    
    def vggt_forward(self, vggt_token_dict: Dict, image_tokens: Tensor = None, image_pos: Tensor = None):
        # image_tokens: [B * S, V, P_image, D]
        # image_pos: [B * S, V, P_image, 2]
        time0 = time.time()

        ### vggt_token_dict -> spatial_tokens_list
        if image_tokens == None:
            image_tokens = vggt_token_dict['image_tokens']
            image_pos = vggt_token_dict['image_tokens_pos']
        spatial_pos = vggt_token_dict['spatial_tokens_pos'] # [B * S, V, P, 2]
        time1 = time.time()
        # logger.debug(f"VGGT encoding time: {time1-time0:.5f} seconds")

        if self.only_2d:
            scene_tokens = image_tokens
            scene_pos = image_pos
        elif self.injector:
        ### spatial_tokens_list, scene_tokens -> INJECTOR -> scene_tokens (2D,3D fusion)
            spatial_tokens_list = [
                vggt_token_dict['spatial_tokens_list'][idx] 
                for idx in self.intermediate_layer_idx
            ]  # [len(layer_idx), B * S, V, P, D_scene]
            spatial_tokens_list = torch.stack(spatial_tokens_list, dim=0)
            camera_tokens_list = vggt_token_dict.get('camera_tokens_list', None)
            if camera_tokens_list != None:
                camera_tokens_list = [
                    camera_tokens_list[idx]
                    for idx in self.intermediate_layer_idx
                ]  # [len(layer_idx), B * S, V, D_camera]
                camera_tokens = camera_tokens_list[-1]
            else:
                camera_tokens = None

            
            if self.view_as_seq:
                L, BS, V, P, D = spatial_tokens_list.shape
                spatial_tokens_list = spatial_tokens_list.view(L, -1, self.view_fake, P, D) # shape: [4, B * S * V, 1, P, D3d]
                spatial_pos = spatial_pos.view(-1, self.view_fake, P, 2) # shape: [B * S * V, 1, P, 2]
                image_tokens = image_tokens.view(-1, self.view_fake, P, D) # shape: [B * S * V, 1, P, D2d]
                image_pos = image_pos.view(-1, self.view_fake, P, 2) # shape: [B * S * V, 1, P, 2]
                scene_tokens = self.injector(
                    spatial_tokens_list, 
                    image_tokens, 
                    pos_3d = spatial_pos,
                    pos_2d = image_pos
                ) # [B * S * V, P, D_out]
                scene_tokens = scene_tokens.view(BS, V, P, D) # [B * S, V, P, D_out]
                scene_pos = spatial_pos.view(BS, V, P, 2) # [B * S, V, P, 2]
            else:
                if self.only_first_view:
                    spatial_tokens_list = spatial_tokens_list[:, :, 0:1, ...] # shape: [4, B * S, V, P, D3d]
                    spatial_pos = spatial_pos[:, 0:1, ...] # shape: [B * S, V, P, 2]
                    # image_tokens = image_tokens[:,0:1,...]
                scene_tokens = self.injector(
                    spatial_tokens_list, 
                    image_tokens, 
                    pos_3d = spatial_pos,
                    pos_2d = image_pos
                ) # [B * S, V, P, D_out]
                scene_pos = spatial_pos
        else:
            ### spatial_tokens_list -> PMHEAD -> scene_tokens (3D fusion)
            spatial_tokens_list = vggt_token_dict['spatial_tokens_list']
            # scene_tokens = spatial_tokens_list[-1]
            scene_tokens, pts3d, pts3d_conf = self.vggt_heads.point_head(
                spatial_tokens_list, pos=spatial_pos, patch_start_idx=0
            ) # [BS, V, D, H', W']
            H_T, W_T = scene_tokens.shape[-2:]
            spatial_pos = torch.stack(
                (torch.meshgrid(torch.arange(H_T), 
                                torch.arange(W_T))), 
                dim=-1
            ).to(spatial_pos.device)
            scene_tokens = scene_tokens.view((*scene_tokens.shape[:3],-1)).permute((0,1,3,2))
            scene_pos = spatial_pos
        time2 = time.time()
        logger.debug(f"Injector time: {time2-time1:.5f} seconds")

        if self.return_predict:
            pts3d = torch.cat((pts3d,pts3d_conf[...,None]),dim=-1)
            return pts3d.view((pts3d.shape[0], -1, 4))
        
        ### scene_tokens -> Multi-View ADAPTER -> scene_features (multi-view fusion)
        if self.mv_fuser != None:
            scene_tokens = self.mv_fuser(scene_tokens, scene_pos, camera_tokens)
            scene_pos = scene_pos[:,0:1,...] # [B, 1, P, 2]

        ### scene_tokens -> MODEL ADAPTER -> scene_features (VA/VLA visual input alignment)
        scene_features = scene_tokens # defalut
        if self.model_adapter != None:
            adapter_input = {
                'x': scene_tokens,
                'xpos': scene_pos,
                'y': None,
                'ypos': None,
            }
            scene_features = self.model_adapter(**adapter_input)
        time3 = time.time()
        logger.debug(f"Model adapter time: {time3-time2:.5f} seconds")
        
        return scene_features

    def forward(
        self, 
        images: Tensor = None, 
        image_tokens: Tensor = None, 
        image_pos: Tensor = None, 
        vggt_token_dict: Dict = None,
        batch_size = None
    ):
        time0 = time.time()
        self.view_fake = None
        if images!=None:
            if len(images.shape) > 5: # images: [B, S, V, 3, H, W]
                batch_size = images.shape[0]
                images = images.view((-1, *images.shape[-4:])) # images: [B * S, V, 3, H, W]
            BS, V, _, H, W = images.shape
            if self.seq_as_view:
                assert batch_size
                self.view_fake = BS * V // batch_size
            elif self.view_as_seq:
                self.view_fake = 1
                
            scene_features = self.image_forward(images, image_tokens, image_pos)
        elif vggt_token_dict:
            if self.view_as_seq:
                self.view_fake = 1
                
            scene_features = self.vggt_forward(vggt_token_dict, image_tokens, image_pos)
        else:
            raise
        time1 = time.time()
        logger.debug(f"Full encoding time: {time1-time0:.5f} seconds")
        return scene_features
    
    
    # TODO: 融合后的token可视化
    # TODO: 传入可视化的min,max用于norm
    def visualization(self, images, vggt_token_dict, save_meta = False, save_pcd = True):
        save_dir = os.environ["DEBUG_DIR"]
        os.makedirs(os.path.join(save_dir, 'depth'), exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'pcd'), exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'meta'), exist_ok=True)
        prefix = time.strftime("%Y%m%d_%H%M%S")

        result = self.vggt_heads(images, **vggt_token_dict)
        H,W = result['depth'].shape[-3:-1]
        batch_size = result['depth'].shape[0]
        token_num = result['depth'].shape[1]
        depth = result['depth'].view((batch_size, token_num, H, W, 1))
        depth = torch.concat(torch.concat(depth.unbind(0),1).unbind(0),1)[None,...]
        save_depth(depth, f"{save_dir}/depth/{prefix}_output_depth.png")
        H,W = images.shape[-2:]
        rgb = images.view((batch_size, token_num, 3, H, W))
        rgb = torch.concat(torch.concat(rgb.unbind(0),2).unbind(0),2)[None,...]
        save_images(rgb, f"{save_dir}/depth/{prefix}_output_rgb.png")
        pcd = result['world_points']
        rgb = images
        depth = result['depth']
        # extrinsic, intrinsic = pose_encoding_to_extri_intri(result["pose_enc"], images.shape[-2:])
        pcd = pcd[:,0:1,...]
        rgb = rgb[:,0:1,...]
        for i in range(len(pcd)):
            if save_meta:
                meta_data = {
                    'vggt_token_dict': vggt_token_dict,
                    'rgb': rgb[i], # V, rgb, H, W
                    'pcd': pcd[i], # V, H, W, xyz
                    'depth': depth[i], # V, H, W, d
                    # 'extri': extrinsic[i], # V, ...
                    # 'intri': intrinsic[i], # V, ...
                }
                torch.save(meta_data, os.path.join(save_dir, "meta", f"{prefix}_{i}.pt"))
            if save_pcd:
                m = torch.tensor([
                    [1.,0.,0.],
                    [0.,-1.,0.],
                    [0.,0.,-1.]
                ]).to(device=pcd.device)
                with torch.autocast(enabled=False,device_type=str(pcd.device)):
                    pcd_i = pcd[i]@m
                process_and_save(pcd_i, rgb[i], os.path.join(save_dir, "pcd", f"{prefix}_{i}.ply"), H)
    
import pathlib
import hydra
import os
import yaml
from omegaconf import OmegaConf
from accelerate.logging import get_logger

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)

# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)

def get_camera_config(camera_type):
    camera_config_path = os.path.join(parent_directory, '../configs/_camera_config.yml')

    assert os.path.isfile(camera_config_path), "task config file is missing"

    with open(camera_config_path, 'r', encoding='utf-8') as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    assert camera_type in args, f'camera {camera_type} is not defined'
    return args[camera_type]


@hydra.main(version_base=None,config_path=str(pathlib.Path(__file__).parent.parent.joinpath('configs')))
def test_diffusion_unet_policy(cfg: OmegaConf):
    # resolve immediately so all the ${now:} resolvers
    # will use the same time.
    head_camera_type = cfg.head_camera_type
    head_camera_cfg = get_camera_config(head_camera_type)
    cfg.image_shape = [3, head_camera_cfg['h'], head_camera_cfg['w']]
    cfg.shape_meta.obs.head_cam.shape = [3, head_camera_cfg['h'], head_camera_cfg['w']]
    OmegaConf.resolve(cfg)

    
    policy = hydra.utils.instantiate(cfg.policy)
    pass

if __name__ == "__main__":
    test_diffusion_unet_policy()
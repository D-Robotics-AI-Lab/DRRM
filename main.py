import pathlib
import hydra
from omegaconf import OmegaConf
from accelerate.logging import get_logger


# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)


@hydra.main(version_base=None,config_path=str(pathlib.Path(__file__).parent.joinpath('configs')))
def main(cfg: OmegaConf):
    OmegaConf.resolve(cfg)

    train_func = hydra.utils.get_method(cfg._target_)
    logger = get_logger(__name__)
    train_func(cfg, logger)


if __name__ == "__main__":
    main()
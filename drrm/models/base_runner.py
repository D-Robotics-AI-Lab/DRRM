import os
from typing import Dict

import hydra
from omegaconf import OmegaConf
from transformers import AutoConfig, AutoModel
from drrm.models.base_policy import BasePolicy
from safetensors.torch import load_model

def load_policy(ckp_path, use_ckp_code = True) -> BasePolicy:
    if use_ckp_code:
        policy_model = AutoModel.from_pretrained(ckp_path, trust_remote_code=True)
        # load state dict of normalizer
        load_model(policy_model, os.path.join(ckp_path, "model.safetensors"), strict=False)
    else:
        # get package path from checkpoint config
        # config = AutoConfig.from_pretrained(ckp_path, trust_remote_code=True)
        config = OmegaConf.load(os.path.join(ckp_path, "config.json"))
        ConfigClass = hydra.utils.get_class(config.pkg_map['AutoConfig'])
        PolicyClass = hydra.utils.get_class(config.pkg_map['AutoModel'])
        # reload config by packege class
        config = ConfigClass.from_pretrained(ckp_path)
        policy_model = PolicyClass(config)
        # load state dict of normalizer
        load_model(policy_model, os.path.join(ckp_path, "model.safetensors"), strict=False)
    return policy_model

class BaseRunner:
    def __init__(self, output_dir):
        self.output_dir = output_dir

    def run(self, policy: BasePolicy) -> Dict:
        raise NotImplementedError()

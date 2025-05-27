import re
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_dpmsolver_multistep import \
    DPMSolverMultistepScheduler

from huggingface_hub import PyTorchModelHubMixin
from huggingface_hub.constants import (PYTORCH_WEIGHTS_NAME,
                                       SAFETENSORS_SINGLE_FILE)
from huggingface_hub.file_download import hf_hub_download
from huggingface_hub.utils import EntryNotFoundError, is_torch_available

from .hub_mixin import CompatiblePyTorchModelHubMixin


class BasePolicy(
        nn.Module, 
        CompatiblePyTorchModelHubMixin, 
    ):
    
    # ========= Train  ============
    def compute_loss(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError("compute_loss is not implemented")
    
    # ========= Inference  ============
    def predict_action(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError("predict_action is not implemented")
    
    def forward(self, *args, **kwargs) -> torch.Tensor:
        return self.compute_loss(*args, **kwargs)

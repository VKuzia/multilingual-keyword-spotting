from typing import Optional, Dict, Any

import torch
import torchvision
from torch import nn

from src.models import Model
from src.models.speech_commands.classification import CoreKernel
from src.utils import inspect_keys


class MSWCModel(Model):
    """
    Uses an untrained instance of EfficientNet_b2.
    """

    @staticmethod
    def get_default_optimizer(kernel: nn.Module) -> torch.optim.Optimizer:
        return torch.optim.SGD(kernel.parameters(), lr=0.1)

    @staticmethod
    def get_default_kernel(args: Optional[Dict[str, Any]] = None) -> nn.Module:
        inspect_keys(args, ["output_channels"])
        backbone: nn.Module = torchvision.models.efficientnet_b2(pretrained=False)
        return CoreKernel(backbone, args["output_channels"])

    @staticmethod
    def get_default_loss_function() -> torch.nn.modules.Module:
        return torch.nn.NLLLoss()

from typing import Type

import torch
from torch import nn

from src.models import Model, EfficientNetKernel


class CoreModel(Model):
    """
    The core of a multilingual embedding.
    Uses an untrained instance of EfficientNet_b0.
    """

    @staticmethod
    def get_embedding_class() -> Type[nn.Module]:
        return EfficientNetKernel

    @staticmethod
    def get_loss_function() -> torch.nn.modules.Module:
        return torch.nn.NLLLoss()

from typing import Type

import torch
from torch import nn

from src.models import Model, EfficientNetKernel, CoreModel


class FsEfficientNetKernel(nn.Module):
    """PyTorch model used as a kernel of FewShotModel."""

    def __init__(self, embedding: EfficientNetKernel = EfficientNetKernel()):
        super().__init__()
        self.embedding = embedding
        # self.embedding.output = nn.Identity()
        self.output = nn.Sequential(
            nn.Linear(embedding.pre_output_categories, 2),
            nn.LogSoftmax(dim=1)
        )
        for param in self.embedding.parameters():
            param.requires_grad = False
        for param in self.output.parameters():
            param.requires_grad = True

    def forward(self, x):
        x = self.embedding(x)
        output = self.output(x)
        return output


class FewShotModel(Model):
    """
    Implementation of a few-shot model based on CoreModel embedding
    """

    @staticmethod
    def get_kernel_class() -> Type[nn.Module]:
        return FsEfficientNetKernel

    @staticmethod
    def get_loss_function() -> torch.nn.modules.Module:
        return torch.nn.NLLLoss()

    # @staticmethod
    # def get_default_optimizer(kernel: nn.Module) -> torch.optim.Optimizer:
    #     return torch.optim.SGD(kernel.output.parameters(), lr=0.1)
    #
    # @staticmethod
    # def get_default_kernel(**kwargs) -> nn.Module:
    #     if 'embedding_class' in kwargs.keys():
    #         return FsEfficientNetKernel(kwargs['embedding_class'].get_default_core_kernel(**kwargs))
    #     else:
    #         raise ValueError("Unknown embedding_class")
    #
    # @staticmethod
    # def get_default_loss_function() -> torch.nn.modules.Module:
    #     return torch.nn.NLLLoss()

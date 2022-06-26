from typing import Type

import torch
from torch import nn

from src.models import Model, TransferClassifier


class FsKernel(nn.Module):
    """PyTorch model used as a kernel of FewShotModel."""

    def __init__(self, embedding: TransferClassifier):
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
        a = self.embedding(x)
        output = self.output(a)
        return output


class FewShotModel(Model):
    """
    Implementation of a few-shot model based on CoreModel embedding
    """

    @staticmethod
    def get_kernel_class() -> Type[nn.Module]:
        return FsKernel

    @staticmethod
    def get_loss_function() -> torch.nn.modules.Module:
        return torch.nn.NLLLoss()

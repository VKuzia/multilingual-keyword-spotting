import torch
from torch import nn

from src.models import CoreKernel, CoreModel2
from src.models import Model


class FewShotKernel(nn.Module):
    """PyTorch model used as a kernel of FewShotModel."""

    def __init__(self, core_embedding: CoreKernel):
        super().__init__()
        self.core = core_embedding
        for param in core_embedding.parameters():
            param.requires_grad = False
        self.softmax = nn.Sequential(
            nn.Linear(core_embedding.output_categories, 2),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        x = self.core(x)
        output = self.softmax(x)
        return output


class FewShotModel(Model):
    """
    Implementation of a few-shot model based on CoreModel embedding
    """

    @staticmethod
    def get_default_optimizer(kernel: nn.Module) -> torch.optim.Optimizer:
        return torch.optim.SGD(kernel.parameters(), lr=0.1)

    @staticmethod
    def get_default_kernel(**kwargs) -> nn.Module:
        return FewShotKernel(CoreModel2.get_default_core_kernel(**kwargs))

    @staticmethod
    def get_default_loss_function() -> torch.nn.modules.Module:
        return torch.nn.NLLLoss()

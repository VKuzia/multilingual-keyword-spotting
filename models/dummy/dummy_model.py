import torch.optim
from torch import nn

from models.model import Model


class DummyKernel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_input = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU()
        )
        self.linear_output = nn.Sequential(
            nn.Linear(256, 10),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.linear_input(x)
        x = self.linear_output(x)
        return x


class DummyModel(Model):
    """A model of a 2 layered linear neural network. Is used for architecture construction only."""

    @staticmethod
    def get_default_loss_function() -> torch.nn.modules.Module:
        return torch.nn.NLLLoss()

    @staticmethod
    def get_default_kernel() -> nn.Module:
        return DummyKernel()

    @staticmethod
    def get_default_optimizer(kernel: nn.Module) -> torch.optim.Optimizer:
        return torch.optim.SGD(kernel.parameters(), lr=0.001)

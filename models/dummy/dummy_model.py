from collections import Callable

import torch.optim
from torch import nn

from models.model import Model, ModelInfoTag


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

    def __init__(self, info_tag: ModelInfoTag):
        kernel: nn.Module = DummyKernel()
        optimizer: torch.optim.Optimizer = torch.optim.SGD(kernel.parameters(), lr=0.001)
        loss_function: Callable = torch.nn.NLLLoss()
        super().__init__(kernel, optimizer, loss_function, info_tag)

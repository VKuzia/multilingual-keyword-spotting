from collections import Callable

import torch.optim
import torchvision
from torch import nn

from models.model import Model, ModelInfoTag


class CoreKernel(nn.Module):
    def __init__(self, efficient_net: nn.Module, output_categories: int):
        super().__init__()
        self.efficient_net = efficient_net
        for layer in self.efficient_net.parameters():
            layer.requires_grad = False

        self.relu1 = nn.Sequential(
            nn.Linear(1000, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU()
        )
        self.relu2 = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU()
        )

        self.selu = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.SELU()
        )
        self.output = nn.Sequential(
            nn.Linear(1024, output_categories),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        x = self.efficient_net(x)
        x = self.relu1(x)
        x = self.relu2(x)
        x = self.selu(x)
        return self.output(x)


class CoreModel(Model):
    def __init__(self, path_to_efficient_net: str, info_tag: ModelInfoTag):
        backbone: nn.Module = torchvision.models.efficientnet_b0()
        backbone.load_state_dict(torch.load(path_to_efficient_net))
        kernel: nn.Module = CoreKernel(backbone, 100)
        optimizer: torch.optim.Optimizer = torch.optim.SGD(kernel.parameters(), lr=0.001)
        loss_function: Callable = torch.nn.NLLLoss()
        super().__init__(kernel, optimizer, loss_function, info_tag)

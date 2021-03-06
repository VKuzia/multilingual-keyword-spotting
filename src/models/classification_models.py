from typing import Optional, Type

import torch.optim
from torch import nn

from src.models import Model
from src.models.efficient_net_crutch import single_b0


class EfficientNetKernel(nn.Module):
    """PyTorch model used as a kernel of CoreModel"""

    def __init__(self, efficient_net: nn.Module = single_b0(),
                 output_channels: Optional[int] = None):
        super().__init__()
        self.output_categories = output_channels
        self.output_on = True if output_channels is not None else False
        self.pre_output_categories = 1024
        self.efficient_net = efficient_net

        # changing last layer of efficient net
        self.efficient_net.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features=self.efficient_net.classifier[1].in_features,
                      out_features=2048),
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

        if self.output_on:
            self.output = nn.Sequential(
                nn.Linear(1024, self.output_categories),
                nn.LogSoftmax(dim=1)
            )

    def forward(self, x):
        x = self.efficient_net(x)
        x = self.relu2(x)
        x = self.selu(x)
        if self.output_on:
            return self.output(x)
        else:
            return x


class CoreModel(Model):
    """
    The core of a multilingual embedding.
    Uses an untrained instance of EfficientNet_b0.
    """

    @staticmethod
    def get_kernel_class() -> Type[nn.Module]:
        return EfficientNetKernel

    @staticmethod
    def get_loss_function() -> torch.nn.modules.Module:
        return torch.nn.NLLLoss()

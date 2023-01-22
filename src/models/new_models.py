from abc import abstractmethod
from typing import Any, Dict

import torch

from src.models.efficient_net_crutch import single_b0

import torch.nn as nn


class Module(nn.Module):
    """
    Abstraction for all Modules in project to save additional information on their structure.
    Basically translates constructor parameters into a dictionary to recreate model further.
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def get_dict(self) -> Dict[str, Any]:
        pass


class TotalKernel(Module):

    def get_dict(self) -> Dict[str, Any]:
        return {
            "embedding": self.embedding.get_dict(),
            "head": self.head.get_dict()
        }

    def __init__(self, embedding: Module, head: Module):
        super().__init__()
        self.embedding = embedding
        self.head = head

    def forward(self, X):
        return self.head(self.embedding(X))


class EfficientNetKernel(Module):

    def get_dict(self) -> Dict[str, Any]:
        return {
            "hidden": self.hidden,
            "output": self.output
        }

    def __init__(self, output: int = 1024, hidden: int = 2048,
                 efficient_net: nn.Module = single_b0()):
        super().__init__()
        self.efficient_net = efficient_net
        self.hidden = hidden
        self.output = output

        # changing last layer of efficient net
        self.efficient_net.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features=self.efficient_net.classifier[1].in_features,
                      out_features=hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU()
        )

        self.relu2 = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU()
        )

        self.selu = nn.Sequential(
            nn.Linear(hidden, output),
            nn.BatchNorm1d(output),
            nn.SELU()
        )

    def forward(self, x):
        x = self.efficient_net(x)
        x = self.relu2(x)
        x = self.selu(x)
        return x


class CnnYKernel(Module):
    """PyTorch model used as a kernel of CoreModel"""

    def get_dict(self) -> Dict[str, Any]:
        return {
            "output": self.output,
            "size": self.size
        }

    def __init__(self, output: int, size: int = 4):
        super().__init__()
        self.output = output
        self.size = size
        self.conv_a_1 = nn.Sequential(
            nn.Conv2d(1, size * 16, kernel_size=(1, 5), stride=(1, 2)),
            nn.BatchNorm2d(size * 16),
            nn.ReLU(),
        )

        self.conv_a_2 = nn.Sequential(
            nn.Conv2d(size * 16, size * 24, kernel_size=(2, 3), stride=(2, 2)),
            nn.BatchNorm2d(size * 24),
            nn.ReLU(),
        )

        self.conv_a_3 = nn.Sequential(
            nn.Conv2d(size * 24, size * 32, kernel_size=(2, 3), stride=(2, 2)),
            nn.BatchNorm2d(size * 32),
            nn.ReLU(),
        )

        self.conv_a_4 = nn.Sequential(
            nn.Conv2d(size * 32, size * 64, kernel_size=(2, 3), stride=(2, 2)),
            nn.BatchNorm2d(size * 64),
            nn.ReLU()
        )

        self.conv_a_5 = nn.Sequential(
            nn.Conv2d(size * 64, size * 192, kernel_size=(5, 2), stride=(1, 1)),
            nn.BatchNorm2d(size * 192),
            nn.ReLU()
        )

        # /////////

        self.conv_b_1 = nn.Sequential(
            nn.Conv2d(1, size * 16, kernel_size=(5, 1), stride=(2, 1)),
            nn.BatchNorm2d(size * 16),
            nn.ReLU(),
        )

        self.conv_b_2 = nn.Sequential(
            nn.Conv2d(size * 16, size * 24, kernel_size=(3, 2), stride=(2, 2)),
            nn.BatchNorm2d(size * 24),
            nn.ReLU()
        )

        self.conv_b_3 = nn.Sequential(
            nn.Conv2d(size * 24, size * 32, kernel_size=(3, 2), stride=(2, 2)),
            nn.BatchNorm2d(size * 32),
            nn.ReLU()
        )

        self.conv_b_4 = nn.Sequential(
            nn.Conv2d(size * 32, size * 64, kernel_size=(3, 2), stride=(2, 2)),
            nn.BatchNorm2d(size * 64),
            nn.ReLU()
        )
        self.conv_b_5 = nn.Sequential(
            nn.Conv2d(size * 64, size * 128, kernel_size=(1, 6), stride=(1, 1)),
            nn.BatchNorm2d(size * 128),
            nn.ReLU()
        )

        self.conv_c_1 = nn.Sequential(
            nn.Conv2d(1, size * 8, kernel_size=(5, 5), stride=(2, 2)),
            nn.BatchNorm2d(size * 8),
            nn.ReLU()
        )

        self.conv_c_2 = nn.Sequential(
            nn.Conv2d(size * 8, size * 16, kernel_size=(3, 3), stride=(2, 2)),
            nn.BatchNorm2d(size * 16),
            nn.ReLU()
        )

        self.conv_c_3 = nn.Sequential(
            nn.Conv2d(size * 16, size * 32, kernel_size=(3, 3), stride=(2, 2)),
            nn.BatchNorm2d(size * 32),
            nn.ReLU()
        )

        self.conv_c_4 = nn.Sequential(
            nn.Conv2d(size * 32, size * 64, kernel_size=(3, 3), stride=(1, 1)),
            nn.BatchNorm2d(size * 64),
            nn.ReLU()
        )

        self.conv_c_5 = nn.Sequential(
            nn.Conv2d(size * 64, size * 96, kernel_size=(1, 3), stride=(1, 1)),
            nn.BatchNorm2d(size * 96),
            nn.ReLU()
        )

        self.linear_1 = nn.Sequential(
            nn.Linear(size * (192 + 128 + 96), output),
            nn.BatchNorm1d(output),
            nn.ReLU(),
            nn.Dropout(p=0.4)
        )

        self.linear_2 = nn.Sequential(
            nn.Linear(output, output),
            nn.SELU(),
            nn.Dropout(p=0.1)
        )

    def forward(self, x):
        one_shaped = x.shape[0] == 1
        if one_shaped:
            print('ONE SHAPED')
            x = torch.stack([x[0], x[0]])
        a = self.conv_a_1(x)
        a = self.conv_a_2(a)
        a = self.conv_a_3(a)
        a = self.conv_a_4(a)
        a = self.conv_a_5(a)
        a = a.view(a.size(0), -1)
        b = self.conv_b_1(x)
        b = self.conv_b_2(b)
        b = self.conv_b_3(b)
        b = self.conv_b_4(b)
        b = self.conv_b_5(b)
        b = b.view(b.size(0), -1)
        c = self.conv_c_1(x)
        c = self.conv_c_2(c)
        c = self.conv_c_3(c)
        c = self.conv_c_4(c)
        c = self.conv_c_5(c)
        c = c.view(c.size(0), -1)
        y = torch.concat([a, b, c], dim=1)
        y = self.linear_1(y)
        y = self.linear_2(y)
        if one_shaped:
            y = y[0].unsqueeze(0)
        return y


class SoftmaxHeadKernel(Module):
    def get_dict(self) -> Dict[str, Any]:
        return {
            "input": self.input,
            "output": self.output
        }

    def __init__(self, input: int, output: int):
        super().__init__()
        self.input = input
        self.output = output
        self.linear = nn.Linear(input, output)
        self.smooth = nn.LogSoftmax(dim=1)

    def forward(self, x):
        return self.smooth(self.linear(x))

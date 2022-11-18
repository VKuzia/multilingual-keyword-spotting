from abc import abstractmethod
from typing import Any, Dict
from src.models.efficient_net_crutch import single_b0

import torch.nn as nn


class Module(nn.Module):

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

import torch.optim
import torchvision
from torch import nn

from src.models.model import Model


class CoreKernel(nn.Module):
    """PyTorch model used as a kernel of CoreModel"""

    def __init__(self, efficient_net: nn.Module, output_categories: int):
        super().__init__()
        self.output_categories = output_categories
        self.efficient_net = efficient_net

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
            nn.Linear(1024, self.output_categories),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        x = self.efficient_net(x)
        x = self.relu1(x)
        x = self.relu2(x)
        x = self.selu(x)
        return self.output(x)


class CoreModel(Model):
    """
    The core of a multilingual embedding.
    Uses an untrained instance of EfficientNet_b0.
    """

    @staticmethod
    def get_default_optimizer(kernel: nn.Module) -> torch.optim.Optimizer:
        return torch.optim.SGD(kernel.parameters(), lr=0.1)

    @staticmethod
    def get_default_kernel() -> nn.Module:
        return CoreModel.get_default_core_kernel()

    @staticmethod
    def get_default_loss_function() -> torch.nn.modules.Module:
        return torch.nn.NLLLoss()

    @staticmethod
    def get_default_core_kernel() -> CoreKernel:
        backbone: nn.Module = torchvision.models.efficientnet_b0(pretrained=False)
        return CoreKernel(backbone, 21)


class CoreModel2(Model):
    """
    The core of a multilingual embedding.
    Uses an untrained instance of EfficientNet_b2.
    """

    @staticmethod
    def get_default_optimizer(kernel: nn.Module) -> torch.optim.Optimizer:
        return torch.optim.SGD(kernel.parameters(), lr=0.1)

    @staticmethod
    def get_default_kernel() -> nn.Module:
        backbone: nn.Module = torchvision.models.efficientnet_b2(pretrained=False)
        return CoreKernel(backbone, 21)

    @staticmethod
    def get_default_loss_function() -> torch.nn.modules.Module:
        return torch.nn.NLLLoss()

    @staticmethod
    def get_default_core_kernel() -> CoreKernel:
        backbone: nn.Module = torchvision.models.efficientnet_b2(pretrained=False)
        return CoreKernel(backbone, 21)

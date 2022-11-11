from abc import abstractmethod
from typing import Optional, Type

import torch.optim
from torch import nn

from src.models import Model
from src.models.efficient_net_crutch import single_b0


class TransferClassifier(nn.Module):

    def __init__(self, pre_output_categories: int):
        super().__init__()
        self.pre_output_categories = pre_output_categories


class LogSoftmaxOutput(nn.Module):
    def __init__(self, pre_output_categories: int, output_categories: int):
        super().__init__()
        self.categories_count = output_categories
        self.linear = nn.Linear(pre_output_categories, output_categories)
        self.smooth = nn.LogSoftmax(dim=1)

    def forward(self, x):
        return self.smooth(self.linear(x))


class ClassifierKernel(nn.Module):

    def __init__(self, embedding: TransferClassifier, output):
        super().__init__()
        self.embedding = embedding
        self.output = output

    def forward(self, x):
        x = self.embedding(x)
        return self.output(x)


class EfficientNetKernel(TransferClassifier):
    """PyTorch model used as a kernel of CoreModel"""

    def __init__(self, efficient_net: nn.Module = single_b0()):
        super().__init__(1024)
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

    def forward(self, x):
        x = self.efficient_net(x)
        x = self.relu2(x)
        x = self.selu(x)
        return x


class CnnKernel(TransferClassifier):
    """PyTorch model used as a kernel of CoreModel"""

    def __init__(self, output_channels: Optional[int] = None, input_shape=(40, 49)):
        super().__init__(2048)
        self.output_categories = output_channels
        self.output_on = True if output_channels is not None else False
        h, w = input_shape

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(2, 2), stride=(1, 1)),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        h -= 1
        w -= 1

        # (h - 1, w - 1)
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(2, 2), stride=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        h -= 1
        w -= 1

        # (h - 2, w - 2)
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 128, kernel_size=(2, 2), stride=(2, 2)),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        h //= 2
        w //= 2

        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 192, kernel_size=(2, 2), stride=(2, 2)),
            nn.BatchNorm2d(192),
            nn.ReLU()
        )
        h //= 2
        w //= 2

        self.conv5 = nn.Sequential(
            nn.Conv2d(192, 256, kernel_size=(2, 2), stride=(2, 2)),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        h //= 2
        w //= 2

        self.conv6 = nn.Sequential(
            nn.Conv2d(256, 384, kernel_size=(2, 2), stride=(2, 2)),
            nn.BatchNorm2d(384),
            nn.ReLU()
        )
        h //= 2
        w //= 2

        self.conv7 = nn.Sequential(
            nn.Conv2d(384, 512, kernel_size=(2, 2), stride=(2, 2)),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        h //= 2
        w //= 2

        self.linear1 = nn.Sequential(
            nn.Linear(512 * w * h, 2048),
            nn.SELU(),
            nn.Dropout(p=0.4)
        )

        self.linear3 = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.SELU(),
            nn.Dropout(p=0.2)
        )

        if self.output_on:
            self.output = nn.Sequential(
                nn.Linear(1024, self.output_categories),
                nn.LogSoftmax(dim=1)
            )

        self.conv_seq = nn.Sequential(
            self.conv1, self.conv2, self.conv3, self.conv4, self.conv5, self.conv6, self.conv7
        )

    def forward(self, x):
        # x = self.conv1(x)
        # x = self.conv2(x)
        # x = self.conv3(x)
        # x = self.conv4(x)
        # x = self.conv5(x)
        x = self.conv_seq(x)
        # print(x.shape)
        x = x.view(x.size(0), -1)
        # print(x.shape)
        x = self.linear1(x)
        # x = self.linear2(x)
        x = self.linear3(x)
        if self.output_on:
            return self.output(x)
        else:
            return x


class CnnModel(Model):
    """
    The core of a multilingual embedding.
    Uses an untrained instance of EfficientNet_b0.
    """

    @staticmethod
    def get_embedding_class() -> Type[nn.Module]:
        return CnnKernel

    @staticmethod
    def get_loss_function() -> torch.nn.modules.Module:
        return torch.nn.NLLLoss()


class CnnXKernel(TransferClassifier):
    """PyTorch model used as a kernel of CoreModel"""

    def __init__(self, output_channels: Optional[int] = None):
        super().__init__(2048)
        self.output_categories = output_channels
        self.output_on = True if output_channels is not None else False

        self.conv_a_1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(1, 5), stride=(1, 2)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        self.conv_a_2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(1, 3), stride=(1, 2)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(2, 1))
        )

        self.conv_a_3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(1, 3), stride=(1, 2)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(2, 1))
        )

        self.conv_a_4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(1, 3), stride=(1, 2)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(2, 1))
        )
        self.conv_a_5 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=(5, 2), stride=(1, 1)),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )

        # /////////

        self.conv_b_1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(5, 1), stride=(2, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        self.conv_b_2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(3, 1), stride=(2, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1, 2))
        )

        self.conv_b_3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3, 1), stride=(2, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1, 2))
        )

        self.conv_b_4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(3, 1), stride=(2, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1, 2))
        )
        self.conv_b_5 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=(1, 6), stride=(1, 1)),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )

        self.conv_c_1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(5, 5), stride=(2, 2)),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        self.conv_c_2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2)),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.conv_c_3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.conv_c_4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self.conv_c_5 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1)),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )

        self.conv_c_6 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=(2, 5), stride=(1, 1)),
            nn.BatchNorm2d(1024),
            nn.ReLU()
        )

        self.linear_1 = nn.Sequential(
            nn.Linear(2048, 1532),
            nn.BatchNorm1d(1532),
            nn.ReLU(),
            nn.Dropout(p=0.3)
        )

        self.linear_2 = nn.Sequential(
            nn.Linear(1532, 1024),
            nn.SELU()
        )

        if self.output_on:
            self.output = nn.Sequential(
                nn.Linear(1024, self.output_categories),
                nn.LogSoftmax(dim=1)
            )

    def forward(self, x):
        # print(x.shape)

        a = self.conv_a_1(x)
        # print(a.shape)
        a = self.conv_a_2(a)
        # print(a.shape)
        a = self.conv_a_3(a)
        # print(a.shape)
        a = self.conv_a_4(a)
        # print(a.shape)
        a = self.conv_a_5(a)
        # print(a.shape)
        a = a.view(a.size(0), -1)

        # print("????????????????????")

        # print(x.shape)
        b = self.conv_b_1(x)
        # print(b.shape)
        b = self.conv_b_2(b)
        # print(b.shape)
        b = self.conv_b_3(b)
        # print(b.shape)
        b = self.conv_b_4(b)
        # print(b.shape)
        b = self.conv_b_5(b)
        # print(b.shape)

        b = b.view(b.size(0), -1)

        # print("///////////////////")

        c = self.conv_c_1(x)
        c = self.conv_c_2(c)
        c = self.conv_c_3(c)
        c = self.conv_c_4(c)
        c = self.conv_c_5(c)
        c = self.conv_c_6(c)
        c = c.view(c.size(0), -1)
        y = torch.concat([a, b, c], dim=1)

        y = self.linear_1(y)
        y = self.linear_2(y)
        if self.output_on:
            return self.output(y)
        else:
            return x


class CnnXModel(Model):
    """
    The core of a multilingual embedding.
    Uses an untrained instance of EfficientNet_b0.
    """

    @staticmethod
    def get_embedding_class() -> Type[nn.Module]:
        return CnnXKernel

    @staticmethod
    def get_loss_function() -> torch.nn.modules.Module:
        return torch.nn.NLLLoss()


class CnnYKernel(TransferClassifier):
    """PyTorch model used as a kernel of CoreModel"""

    def __init__(self, output_channels: Optional[int] = None):
        super().__init__(1536)
        self.output_categories = output_channels
        self.output_on = True if output_channels is not None else False

        self.conv_a_1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(1, 5), stride=(1, 2)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.conv_a_2 = nn.Sequential(
            nn.Conv2d(64, 96, kernel_size=(2, 3), stride=(2, 2)),
            nn.BatchNorm2d(96),
            nn.ReLU(),
        )

        self.conv_a_3 = nn.Sequential(
            nn.Conv2d(96, 128, kernel_size=(2, 3), stride=(2, 2)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.conv_a_4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(2, 3), stride=(2, 2)),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self.conv_a_5 = nn.Sequential(
            nn.Conv2d(256, 768, kernel_size=(5, 2), stride=(1, 1)),
            nn.BatchNorm2d(768),
            nn.ReLU()
        )

        # /////////

        self.conv_b_1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(5, 1), stride=(2, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.conv_b_2 = nn.Sequential(
            nn.Conv2d(64, 96, kernel_size=(3, 2), stride=(2, 2)),
            nn.BatchNorm2d(96),
            nn.ReLU()
        )

        self.conv_b_3 = nn.Sequential(
            nn.Conv2d(96, 128, kernel_size=(3, 2), stride=(2, 2)),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.conv_b_4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(3, 2), stride=(2, 2)),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.conv_b_5 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=(1, 6), stride=(1, 1)),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )

        self.conv_c_1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(5, 5), stride=(2, 2)),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        self.conv_c_2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2)),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.conv_c_3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2)),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.conv_c_4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self.conv_c_5 = nn.Sequential(
            nn.Conv2d(256, 384, kernel_size=(1, 3), stride=(1, 1)),
            nn.BatchNorm2d(384),
            nn.ReLU()
        )

        # self.conv_c_6 = nn.Sequential(
        #     nn.Conv2d(512, 1024, kernel_size=(2, 5), stride=(1, 1)),
        #     nn.BatchNorm2d(1024),
        #     nn.ReLU()
        # )

        self.linear_1 = nn.Sequential(
            nn.Linear(1664, 1536),
            nn.BatchNorm1d(1536),
            nn.ReLU(),
            nn.Dropout(p=0.4)
        )

        self.linear_2 = nn.Sequential(
            nn.Linear(1536, 1536),
            nn.SELU(),
            nn.Dropout(p=0.1)
        )

        if self.output_on:
            self.output = nn.Sequential(
                nn.Linear(1536, self.output_categories),
                nn.LogSoftmax(dim=1)
            )

    def forward(self, x):
        # print(x.shape)

        a = self.conv_a_1(x)
        # print(a.shape)
        a = self.conv_a_2(a)
        # print(a.shape)
        a = self.conv_a_3(a)
        # print(a.shape)
        a = self.conv_a_4(a)
        # print(a.shape)
        a = self.conv_a_5(a)
        # print(a.shape)
        a = a.view(a.size(0), -1)

        # print("????????????????????")

        # print(x.shape)
        b = self.conv_b_1(x)
        # print(b.shape)
        b = self.conv_b_2(b)
        # print(b.shape)
        b = self.conv_b_3(b)
        # print(b.shape)
        b = self.conv_b_4(b)
        # print(b.shape)
        b = self.conv_b_5(b)
        # print(b.shape)

        b = b.view(b.size(0), -1)

        # print("///////////////////")

        c = self.conv_c_1(x)
        # print(c.shape)
        c = self.conv_c_2(c)
        # print(c.shape)
        c = self.conv_c_3(c)
        # print(c.shape)
        c = self.conv_c_4(c)
        # print(c.shape)
        c = self.conv_c_5(c)
        # print(c.shape)
        # c = self.conv_c_6(c)
        c = c.view(c.size(0), -1)
        y = torch.concat([a, b, c], dim=1)

        y = self.linear_1(y)
        y = self.linear_2(y)
        if self.output_on:
            return self.output(y)
        else:
            return y


class CnnYModel(Model):
    """
    The core of a multilingual embedding.
    Uses an untrained instance of EfficientNet_b0.
    """

    @staticmethod
    def get_embedding_class() -> Type[nn.Module]:
        return CnnYKernel

    @staticmethod
    def get_loss_function() -> torch.nn.modules.Module:
        return torch.nn.NLLLoss()

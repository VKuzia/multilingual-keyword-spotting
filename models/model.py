from dataclasses import dataclass

import torch.optim
from torch import nn


@dataclass
class ModelInfoTag:
    """
    Dataclass containing information on the text description of the model.
    Would be used for saving files, titles of plots etc.
    """
    name: str
    version_tag: str


@dataclass
class ModelLearningInfo:
    """
    Dataclass containing information about model learning progress.
    Contains useful information for printing model's stats.
    """
    epochs_trained: int = 0
    last_loss: float = 0.0


class Model:
    """
    Class wrapping pytorch implementations into a solid abstraction to work with.
    Contains all information needed to be trained and to produce understandable output.
    Provides interface to make predictions as torch.nn.Module does.

    Important note: this class is not responsible for training itself. This logic is moved to trainers.Trainer.

    TODO: dynamic parameters change.
    TODO: loading to/from file.
    """

    def __init__(self, kernel: nn.Module, optimizer: torch.optim.Optimizer, loss_function: torch.nn.modules.Module,
                 info_tag: ModelInfoTag, cuda: bool = True):
        self.kernel: nn.Module = kernel.to('cuda' if cuda else 'cpu')
        self.optimizer: torch.optim.Optimizer = optimizer
        self.loss_function: torch.nn.modules.Module = loss_function
        self.info_tag: ModelInfoTag = info_tag
        self.learning_info: ModelLearningInfo = ModelLearningInfo()

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        """
        Produces model's prediction for input data provided.
        :param data: input data to make prediction on
        :return: output tensor i.e. the prediction
        """
        return self.kernel(data)

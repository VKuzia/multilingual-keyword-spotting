from abc import abstractmethod
from dataclasses import dataclass

import torch.optim
from torch import nn
from typing import Dict, Any, Type


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


@dataclass
class ModelCheckpoint:
    kernel_state_dict: Dict[Any, Any]
    optimizer_state_dict: Dict[Any, Any]
    learning_info: ModelLearningInfo
    info_tag: ModelInfoTag
    id: int


class Model:
    """
    Class wrapping pytorch implementations into a solid abstraction to work with.
    Contains all information needed to be trained and to produce understandable output.
    Provides interface to make predictions as torch.nn.Module does.

    Important note: this class is not responsible for training itself. This logic is moved to trainers.Trainer.

    TODO: dynamic parameters change.
    TODO: loading to/from file.
    """
    optimizer_class: torch.optim.Optimizer
    kernel_class: nn.Module

    def __init__(self, kernel: nn.Module, optimizer: torch.optim.Optimizer, loss_function: torch.nn.modules.Module,
                 info_tag: ModelInfoTag, cuda: bool = True):
        self.kernel: nn.Module = kernel.to('cuda' if cuda else 'cpu')
        self.optimizer: torch.optim.Optimizer = optimizer
        self.loss_function: torch.nn.modules.Module = loss_function
        self.info_tag: ModelInfoTag = info_tag
        self.learning_info: ModelLearningInfo = ModelLearningInfo()
        self.checkpoint_id: int = 0

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        """
        Produces model's prediction for input data provided.
        :param data: input data to make prediction on
        :return: output tensor i.e. the prediction
        """
        return self.kernel(data)

    def build_checkpoint(self) -> ModelCheckpoint:
        self.checkpoint_id += 1
        return ModelCheckpoint(self.kernel.state_dict(),
                               self.optimizer.state_dict(),
                               self.learning_info,
                               self.info_tag,
                               self.checkpoint_id)

    @staticmethod
    @abstractmethod
    def get_default_optimizer(kernel: nn.Module) -> torch.optim.Optimizer:
        pass

    @staticmethod
    @abstractmethod
    def get_default_kernel() -> nn.Module:
        pass

    @staticmethod
    @abstractmethod
    def get_default_loss_function() -> torch.nn.modules.Module:
        pass


def build_model_of(model_class: Type[Model], info_tag: ModelInfoTag, cuda: bool = True) -> Model:
    kernel: nn.Module = model_class.get_default_kernel()
    model: Model = Model(kernel, model_class.get_default_optimizer(kernel),
                         model_class.get_default_loss_function(), info_tag, cuda)
    return model

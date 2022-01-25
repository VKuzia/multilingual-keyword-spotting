from typing import Dict, Any, Type, Optional
from abc import abstractmethod

import torch.optim
from torch import nn

from utils import no_none_dataclass


@no_none_dataclass
class ModelInfoTag:
    """
    Dataclass containing information on the text description of the model.
    Would be used for saving files, titles of plots etc.
    """
    name: str
    version_tag: str


@no_none_dataclass
class ModelLearningInfo:
    """
    Dataclass containing information about model learning progress.
    Contains useful information for printing model's stats.
    """
    epochs_trained: int = 0
    last_loss: float = 0.0


@no_none_dataclass
class ModelCheckpoint:
    """
    Dataclass containing model's state to be saved.
    Is used in loading and saving models. See ModelIOHelper.
    """
    kernel_state_dict: Dict[Any, Any]
    optimizer_state_dict: Dict[Any, Any]
    learning_info: ModelLearningInfo
    info_tag: ModelInfoTag
    checkpoint_id: int


class Model:
    """
    Class wrapping pytorch implementations into a solid abstraction to work with.
    Contains all information needed to be trained and to produce understandable output.
    Provides interface to make predictions as torch.nn.Module does.

    All children of this base must implement abstract methods providing model's kernel,
    optimizer and loss function.

    Important note: this class is not responsible for training itself.
    This logic is moved to trainers.Trainer.
    """

    def __init__(self, kernel: nn.Module, optimizer: torch.optim.Optimizer,
                 loss_function: torch.nn.modules.Module,
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
        """
        Returns model's data in a form of a ModelCheckpoint instance
        :return: ModelCheckpoint instance
        """
        self.checkpoint_id += 1
        return ModelCheckpoint(self.kernel.state_dict(),
                               self.optimizer.state_dict(),
                               self.learning_info,
                               self.info_tag,
                               self.checkpoint_id)

    @staticmethod
    @abstractmethod
    def get_default_optimizer(kernel: nn.Module) -> torch.optim.Optimizer:
        """Returns optimizer to construct initial models of given class with"""

    @staticmethod
    @abstractmethod
    def get_default_kernel(args: Optional[Dict[str, Any]] = None) -> nn.Module:
        """Returns kernel (nn.Module) to construct initial models of given class with"""

    @staticmethod
    @abstractmethod
    def get_default_loss_function() -> torch.nn.modules.Module:
        """Returns loss function to construct models of given class with"""


def build_model_of(model_class: Type[Model], info_tag: ModelInfoTag, *,
                   kernel_args: Optional[Dict[str, Any]] = None,
                   kernel: Optional[nn.Module] = None,
                   optimizer: Optional[torch.optim.Optimizer] = None,
                   cuda: bool = True) -> Model:
    """Returns a default (initial) model of a given class"""
    kernel: nn.Module = kernel if kernel is not None else model_class.get_default_kernel(
        kernel_args)
    optimizer: torch.optim.Optimizer = \
        optimizer if optimizer is not None else model_class.get_default_optimizer(kernel)
    model: Model = Model(kernel, optimizer, model_class.get_default_loss_function(), info_tag, cuda)
    return model

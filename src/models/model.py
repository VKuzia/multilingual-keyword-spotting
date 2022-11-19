from dataclasses import field
from typing import Dict, Any, Type, List
from abc import abstractmethod
from collections import defaultdict

import torch.optim
from torch import nn

import src.models.new_models as new_models
from src.utils import no_none_dataclass


@no_none_dataclass(iterable_ok=True)
class ModelLearningInfo:
    """
    Dataclass containing information about model's learning progress.
    Contains useful information for printing model's stats.
    """
    metrics: Dict[str, List[float]] = field(default_factory=lambda: defaultdict(list))
    other: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metrics": self.metrics,
            "other": self.other
        }


@no_none_dataclass()
class ModelCheckpoint:
    """
    Dataclass containing model's state to be saved.
    Is used in loading and saving models. See ModelIOHelper.
    """
    embedding_state_dict: Dict[Any, Any]
    output_layer_state_dict: Dict[Any, Any]
    optimizer_state_dict: Dict[Any, Any]
    scheduler_state_dict: Dict[Any, Any]
    learning_info: ModelLearningInfo


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

    def __init__(self,
                 kernel: new_models.TotalKernel,
                 optimizer: torch.optim.Optimizer,
                 scheduler,
                 loss_function: torch.nn.modules.Module,
                 cuda: bool = True):
        self.kernel: new_models.TotalKernel = kernel.to('cuda' if cuda else 'cpu')
        self.optimizer: torch.optim.Optimizer = optimizer
        self.scheduler = scheduler
        self.loss_function: torch.nn.modules.Module = loss_function
        self.learning_info: ModelLearningInfo = ModelLearningInfo()

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
        return ModelCheckpoint(self.kernel.embedding.state_dict(),
                               self.kernel.output.state_dict(),
                               self.optimizer.state_dict(),
                               self.scheduler.state_dict(),
                               self.learning_info)

    @staticmethod
    @abstractmethod
    def get_embedding_class() -> Type[nn.Module]:
        """Returns kernel nn.Module of given model"""

    @staticmethod
    @abstractmethod
    def get_loss_function() -> torch.nn.modules.Module:
        """Returns loss function to construct models of given class with"""

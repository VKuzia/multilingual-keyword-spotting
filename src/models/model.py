from dataclasses import field
from typing import Dict, Any, List
from collections import defaultdict

import torch.optim
import src.models.new_models as new_models
from src.utils import no_none_dataclass


@no_none_dataclass(iterable_ok=True)
class ModelLearningInfo:
    """
    Dataclass containing information about model's learning progress.
    Contains useful information for printing model's stats and metrics.
    """
    metrics: Dict[str, List[float]] = field(default_factory=lambda: defaultdict(list))
    other: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metrics": self.metrics,
            "other": self.other
        }


class Model:
    """
    Class wrapping pytorch elements into a solid abstraction to work with.
    Provides interface to make predictions as torch.nn.Module does.
    Stores training information for saving/logging.

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
        return self.kernel(data)

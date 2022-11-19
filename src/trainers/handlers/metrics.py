from enum import Enum
from typing import List, Optional

import torch
import torch.nn.functional as F

from src.dataloaders import DataLoader
from src.models import Model
from src.trainers.handlers import LearningHandler, HandlerMode


class ValidationMode(Enum):
    VALIDATION = 0
    TRAINING = 1


class MetricHandler(LearningHandler):
    """Performs metric estimation on a portion of data provided via data_loader"""

    def __init__(self, data_loader: DataLoader, metrics: List[str], batch_count: int,
                 mode: ValidationMode = ValidationMode.VALIDATION, switch_to_train: bool = True):
        self.data_loader = data_loader
        self.batch_count = batch_count
        self.validation_mode = mode
        self.metrics_names = metrics
        self.switch_to_train = switch_to_train
        self.metric_functions = [self.build_metrics(name) for name in metrics]

    def build_metrics(self, name):
        if name == 'multiacc':
            return estimate_multiclass_accuracy
        if name == 'xent':
            return estimate_cross_entropy_loss
        else:
            raise ValueError(f'Metric "{name}" is unknown.')

    def handle(self, model: Model, mode: HandlerMode = HandlerMode.NONE) -> None:
        model.kernel.eval()
        pred = []
        outputs = []
        target = []
        with torch.no_grad():
            for _ in range(self.batch_count):
                data_batch, labels_batch = self.data_loader.get_batch()
                target.append(labels_batch)
                output = model(data_batch)
                pred.append(output.argmax(dim=1))
                outputs.append(output)
            pred = torch.concat(pred)
            target = torch.concat(target)
            outputs = torch.concat(outputs)
            for name, func in zip(self.metrics_names, self.metric_functions):
                key = f'{name}_{"train" if self.validation_mode == ValidationMode.TRAINING else "val"}'
                model.learning_info.metrics[key].append(func(target, pred, outputs))
        if self.switch_to_train:
            model.kernel.train()


def estimate_multiclass_accuracy(target: torch.Tensor, pred: torch.Tensor,
                                 outputs: Optional[torch.Tensor]) -> float:
    labels = set(target.tolist())
    accuracies = {x: (torch.sum(torch.logical_and(target == x, target == pred))
                     / torch.sum(target == x)).item() for x in labels}
    weights = {x: (torch.sum(target == x) / len(target)).item() for x in labels}
    return sum([accuracies[x] * weights[x] for x in labels])


def estimate_cross_entropy_loss(target: torch.Tensor, pred: torch.Tensor,
                                outputs: Optional[torch.Tensor]) -> float:
    return F.cross_entropy(outputs, target).item()

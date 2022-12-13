from enum import Enum
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
import torchmetrics.classification
from sklearn.metrics import f1_score

from sklearn.metrics import roc_curve
from tqdm import trange

from src.dataloaders import DataLoader
from src.models import Model
from src.trainers.handlers import LearningHandler, HandlerMode


class ValidationMode(Enum):
    VALIDATION = 0
    TRAINING = 1


class MetricHandler(LearningHandler):
    """Performs metric estimation on a portion of data provided via data_loader"""

    def __init__(self, data_loader: DataLoader, metrics: List[str], batch_count: Optional[int],
                 mode: ValidationMode = ValidationMode.VALIDATION, switch_to_train: bool = True):
        self.data_loader = data_loader
        self.batch_count = batch_count if batch_count else data_loader.get_batch_count()
        self.validation_mode = mode
        self.metrics_names = metrics
        self.switch_to_train = switch_to_train
        self.metric_functions = [self.build_metrics(name) for name in metrics]

    def build_metrics(self, name):
        if name == 'multiacc':
            return estimate_multiclass_accuracy
        if name == 'xent':
            return estimate_cross_entropy_loss
        if name == 'f1_score':
            return estimate_weighted_f1_score
        if name == 'binary_f1_score':
            return estimate_binary_f1_score
        if name == 'roc_auc':
            return estimate_roc_auc
        if name == 'precision':
            return estimate_precision
        if name == 'recall':
            return estimate_recall
        if name == 'eer':
            return estimate_eer
        else:
            raise ValueError(f'Metric "{name}" is unknown.')

    def handle(self, model: Model, mode: HandlerMode = HandlerMode.NONE) -> None:
        model.kernel.eval()
        pred = []
        outputs = []
        target = []
        self.data_loader.reset()
        with torch.no_grad():
            for _ in trange(self.batch_count, leave=False, desc='VAL'):
                data_batch, labels_batch = self.data_loader.get_batch()
                target.append(labels_batch.cpu())
                output = model(data_batch).cpu()
                pred.append(output.argmax(dim=1))
                outputs.append(output)
            pred = torch.concat(pred)
            target = torch.concat(target)
            outputs = torch.concat(outputs)
            for name, func in zip(self.metrics_names, self.metric_functions):
                key = f'{name}_{"train" if self.validation_mode == ValidationMode.TRAINING else "val"}'
                result = self._use_func(func, name, target, pred, outputs)
                model.learning_info.metrics[key].append(result)
        if self.switch_to_train:
            model.kernel.train()

    @staticmethod
    def _use_func(func, name, target, pred, outputs):
        if name in {'roc_auc', 'xent', 'eer'}:
            return func(target, outputs)
        if name in {'multiacc', 'f1_score', 'binary_f1_score', 'precision', 'recall'}:
            return func(target, pred)
        raise ValueError(f'Unknown name "{name}"')


def estimate_multiclass_accuracy(target: torch.Tensor, pred: torch.Tensor) -> float:
    labels = set(target.tolist())
    accuracies = {x: (torch.sum(torch.logical_and(target == x, target == pred))
                      / torch.sum(target == x)).item() for x in labels}
    weights = {x: (torch.sum(target == x) / len(target)).item() for x in labels}
    return sum([accuracies[x] * weights[x] for x in labels])


def estimate_cross_entropy_loss(target: torch.Tensor, outputs: torch.Tensor) -> float:
    return F.cross_entropy(outputs, target).item()


def estimate_roc_auc(target: torch.Tensor, outputs: torch.Tensor) -> float:
    func = torchmetrics.classification.BinaryAUROC()
    return func(outputs[:, 1], target).item()


def estimate_eer(target: torch.Tensor, outputs: torch.Tensor) -> float:
    fpr, tpr, threshold = roc_curve(target, outputs[:, 1], pos_label=1)
    fnr = 1 - tpr
    # eer_threshold = threshold[np.nanargmin(np.absolute((fnr - fpr)))]
    eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    return eer


def estimate_precision(target: torch.Tensor, preds: torch.Tensor) -> float:
    func = torchmetrics.classification.Precision(task='binary')
    return func(preds, target).item()


def estimate_recall(target: torch.Tensor, preds: torch.Tensor) -> float:
    func = torchmetrics.classification.Recall(task='binary')
    return func(preds, target).item()


def estimate_binary_f1_score(target: torch.Tensor, preds: torch.Tensor) -> float:
    return f1_score(target, preds, average='binary')


def estimate_weighted_f1_score(target: torch.Tensor, preds: torch.Tensor) -> float:
    return f1_score(target, preds, average='weighted')

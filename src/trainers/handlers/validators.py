from collections import defaultdict
from enum import Enum
from typing import Tuple, Dict

import torch
from tqdm import trange

from src.dataloaders import DataLoader
from src.models import Model
from src.trainers.handlers import LearningHandler, HandlerMode


class ValidationMode(Enum):
    VALIDATION = 0
    TRAINING = 1


class ClassificationValidator(LearningHandler):
    """Performs accuracy estimation on a set of data provided via data_loader"""

    def __init__(self, data_loader: DataLoader, batch_count: int,
                 mode: ValidationMode = ValidationMode.VALIDATION, switch_to_train: bool = True):
        self.data_loader = data_loader
        self.batch_count = batch_count
        self.validation_mode = mode
        self.switch_to_train = switch_to_train

    def handle(self, model: Model, mode: HandlerMode = HandlerMode.NONE) -> None:
        """
        Performs self.batch_count tests accumulating number of correct answers
        and writes the final accuracy to self.output_stream
        """
        model.kernel.eval()
        accuracy = estimate_accuracy(model, self.data_loader, self.batch_count)
        if self.validation_mode.value == ValidationMode.VALIDATION.value:
            model.learning_info.val_accuracy_history.append(accuracy)
        elif self.validation_mode.value == ValidationMode.TRAINING.value:
            model.learning_info.train_accuracy_history.append(accuracy)
        if self.switch_to_train:
            model.kernel.train()


def estimate_accuracy(model: Model, data_loader: DataLoader, batch_count: int) -> float:
    """
    Calculates accuracy for given model on given data_loader.
    Accumulates number of correct answers in batch_count batches and
    returns the sum divided by total number of samples.
    :param model: model to be estimated
    :param data_loader: data source to perform estimation on
    :param batch_count: number of batches to use for estimation
    :return: accuracy float value from 0 to 1
    """
    correct: int = 0
    total: int = 0
    for _ in range(batch_count):
        data_batch, labels_batch = data_loader.get_batch()
        model_output = model(data_batch).argmax(dim=1)
        # gpu-cpu sync: performance bottleneck
        # do not use while training
        correct += torch.sum(model_output == labels_batch).item()
        total += len(model_output)
    return correct / total


def estimate_accuracy_with_errors(model: Model, data_loader: DataLoader, batch_count: int) \
        -> Tuple[float, Dict[Tuple[int, int], int]]:
    model.kernel.eval()
    correct: int = 0
    total: int = 0
    results: Dict[Tuple[int, int], int] = defaultdict(lambda: 0)
    for _ in trange(batch_count):
        data_batch, labels_batch = data_loader.get_batch()
        model_output = model(data_batch).argmax(dim=1)
        # gpu-cpu sync: performance bottleneck
        # do not use while training
        for model_result, labels_result in zip(model_output, labels_batch):
            if model_result == labels_result:
                continue
            results[(model_result.item(), labels_result.item())] += 1
        correct += torch.sum(model_output == labels_batch).item()
        total += len(model_output)
    return correct / total, results

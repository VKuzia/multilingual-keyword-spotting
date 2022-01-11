import sys
from typing import IO

import torch

from src.dataloaders import DataLoader
from src.models import Model
from src.trainers.handlers import LearningHandler, HandlerMode


class ClassificationValidator(LearningHandler):
    """Performs accuracy estimation on a set of data provided via data_loader"""

    def __init__(self, data_loader: DataLoader, batch_count: int,
                 output_stream: IO[str] = sys.stdout):
        self.data_loader = data_loader
        self.batch_count = batch_count
        self.output_stream = output_stream

    def handle(self, model: Model, mode: HandlerMode = HandlerMode.NONE) -> None:
        """
        Performs self.batch_count tests accumulating number of correct answers
        and writes the final accuracy to self.output_stream
        """
        accuracy = estimate_accuracy(model, self.data_loader, self.batch_count)
        self.output_stream.write("Accuracy on validation set: {:.5f}\n".format(accuracy))
        self.output_stream.flush()


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
        correct += torch.sum(model_output == labels_batch).item()
        total += len(model_output)
    return correct / total

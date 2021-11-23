import sys
from abc import ABC, abstractmethod
import time
from enum import Enum
from typing import Optional

import torch
from typing.io import IO

from dataloaders.dataloader import DataLoader
from models.model import Model
from models.model_loader import ModelIOHelper


class HandlerMode(Enum):
    """
    Modes of LearningHandler children representing the time they are used in.
    """
    NONE = -1
    PRE_EPOCH = 0
    STEP = 1
    POST_EPOCH = 2


class LearningHandler(ABC):
    """
    The base class for different task handlers used while training the model.
    Basically, its children provide a simple method which changes/reads the model
    and perform some task according to the data they are given and current HandlerMode.
    """

    @abstractmethod
    def handle(self, model: Model, mode: HandlerMode = HandlerMode.NONE) -> None:
        """
        Performs the handlers job. May change/read from model.
        :param model: the Model instance to get information from or perform task on.
        :param mode: current stage of learning, is used to perform different tasks in different stages of learning
        without need to share the data elsewhere but inside the LearningHandler
        :return: None.
        :raises: ValueError if the task can't be performed with current mode
        """
        pass


class TimeEpochHandler(LearningHandler):
    """
    Implementation of LearningHandler which is used to measure time elapsed within one epoch training.
    To be correctly used needs to be both in pre_epoch_handlers and post_epoch_handlers (see trainers.Trainer).
    Should be the last in pre_epoch_handlers and first in post_epoch_handlers to measure train time without
    other handlers jobs.
    """

    def __init__(self, output_stream: IO[str] = sys.stdout):
        self.start_time: Optional[float] = None
        self.output_stream: IO[str] = output_stream

    def handle(self, model: Model, mode: HandlerMode = HandlerMode.NONE) -> None:
        if mode == HandlerMode.PRE_EPOCH:
            self.handle_pre_epoch_job()
        elif mode == HandlerMode.POST_EPOCH:
            self.handle_post_epoch_job()
        else:
            raise ValueError("Unknown HandlerMode.")

    def handle_pre_epoch_job(self) -> None:
        """Stores time of epoch learning start"""
        self.start_time = time.time()

    def handle_post_epoch_job(self) -> None:
        """
        Calculates time elapsed while training last epoch.
        handle_pre_epoch_job call must be performed before handle_post_epoch_job.
        :raises: TypeError if self.start_time is None, meaning start time is not measured properly.
        :return: None
        """
        if self.start_time is None:
            raise TypeError("Couldn't measure time as start time is None.")
        delta: float = time.time() - self.start_time
        self.output_stream.write("Time for epoch: {:.4f}\n".format(delta))
        self.output_stream.flush()
        self.start_time = None


class StepLossHandler(LearningHandler):
    """
    Implementation of LearningHandler which is used to output last model's loss
    or other meaningful information on one step of learning.
    """

    def __init__(self, output_stream: IO[str] = sys.stdout):
        self.output_stream = output_stream

    def handle(self, model: Model, mode: HandlerMode = HandlerMode.NONE) -> None:
        """Writes the last model's lost to self.output_stream"""
        self.output_stream.write("[{}] Loss for batch: {:.6f}\n"
                                 .format(model.learning_info.epochs_trained + 1, model.learning_info.last_loss))
        self.output_stream.flush()


class ClassificationValidator(LearningHandler):
    """Performs accuracy estimation on a set of data provided via data_loader"""

    def __init__(self, data_loader: DataLoader, batch_count: int, batch_size: int, output_stream: IO[str] = sys.stdout):
        self.data_loader = data_loader
        self.batch_count = batch_count
        self.batch_size = batch_size
        self.output_stream = output_stream

    def handle(self, model: Model, mode: HandlerMode = HandlerMode.NONE) -> None:
        """
        Performs self.batch_count tests accumulating number of correct answers
        and writes the final accuracy to self.output_stream
        """
        correct: int = 0
        for batch_num in range(self.batch_count):
            data_batch, labels_batch = self.data_loader.get_batch(self.batch_size)
            model_output = model(data_batch).argmax(dim=1)
            correct += torch.sum(model_output == labels_batch)
        accuracy = correct / (self.batch_count * self.batch_size)
        self.output_stream.write("Accuracy on validation set: {:.5f}\n".format(accuracy))
        self.output_stream.flush()


class ModelSaver(LearningHandler):
    """
    Saves the model one time in epoch_rate epochs. Uses ModelIOHelper instance provided.
    """

    def __init__(self, model_io: ModelIOHelper, epoch_rate: int, path: Optional[str] = None,
                 use_base_path: bool = True):
        self.model_io = model_io
        self.epoch_rate = epoch_rate
        self.epochs_to_save = epoch_rate
        self.path: Optional[str] = path
        self.use_base_path: bool = use_base_path

    def handle(self, model: Model, mode: HandlerMode = HandlerMode.NONE) -> None:
        """Decreases self.epochs_to_save and if it's time to save the model, uses self.model_io to do it"""
        self.epochs_to_save -= 1
        if self.epochs_to_save == 0:
            self.epochs_to_save = self.epoch_rate
            self.model_io.save_model(model, self.path, self.use_base_path)

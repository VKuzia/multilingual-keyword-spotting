import os
from abc import ABC, abstractmethod
from enum import Enum

from src.models import Model, ModelIO
import src.dataloaders as data


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
        :param mode: current stage of learning, used to perform tasks in distinct stages of learning
        without need to share the data elsewhere but inside the LearningHandler
        :return: None.
        :raises: ValueError if the task can't be performed with current mode
        """


class ModelSaver(LearningHandler):
    """
    Saves the model one time in epoch_rate epochs. Uses ModelIOHelper instance provided.
    """

    def __init__(self, model_io: ModelIO, config, output_dir: str,
                 epoch_rate: int = 1, full_path: bool = True):
        self.model_io = model_io
        self.epoch_rate = epoch_rate
        self.epochs_to_save = epoch_rate
        self.config = config
        self.output_dir = output_dir
        self.full_path = full_path
        self.version = 0

    def handle(self, model: Model, mode: HandlerMode = HandlerMode.NONE) -> None:
        """Decreases self.epochs_to_save and uses self.model_io to save model when needed"""
        self.epochs_to_save -= 1
        if self.epochs_to_save == 0:
            self.version += 1
            self.epochs_to_save = self.epoch_rate
            self.model_io.save_model(self.config, model,
                                     os.path.join(self.output_dir, str(self.version)),
                                     full_path=self.full_path)


class DataloaderResetter(LearningHandler):

    def __init__(self, loader: data.DataLoader):
        self.loader = loader

    def handle(self, model: Model, mode: HandlerMode = HandlerMode.NONE):
        self.loader.reset()

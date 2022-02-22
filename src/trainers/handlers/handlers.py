from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional

from src.models import Model, ModelIOHelper


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

    def __init__(self, model_io: ModelIOHelper, epoch_rate: int = 1, path: Optional[str] = None,
                 use_base_path: bool = True):
        self.model_io = model_io
        self.epoch_rate = epoch_rate
        self.epochs_to_save = epoch_rate
        self.path: Optional[str] = path
        self.use_base_path: bool = use_base_path

    def handle(self, model: Model, mode: HandlerMode = HandlerMode.NONE) -> None:
        """Decreases self.epochs_to_save and uses self.model_io to save model when needed"""
        self.epochs_to_save -= 1
        if self.epochs_to_save == 0:
            self.epochs_to_save = self.epoch_rate
            self.model_io.save_model(model, self.path, self.use_base_path)

from abc import abstractmethod
from typing import List

from src.dataloaders import DataLoader
from src.trainers.handlers import LearningHandler, HandlerMode
from src.models import Model
from src.utils import no_none_dataclass


@no_none_dataclass()
class TrainingParams:
    """
    Dataclass containing parameters used in a network training loop.
    Important note: this class is not responsible for model's hyperparameters.
    """
    batch_size: int
    batch_count: int
    epoch_count: int
    cuda: bool = True


class Trainer:
    """
    Base class for performing the training cycle of a model.
    Provides an interface to train the model with given lists of handlers.
    """

    def __init__(self, pre_epoch_handlers: List[LearningHandler] = None,
                 after_step_handlers: List[LearningHandler] = None,
                 post_epoch_handlers: List[LearningHandler] = None,
                 silent: bool = False):
        self.pre_epoch_handlers = pre_epoch_handlers
        self.after_step_handlers = after_step_handlers
        self.post_epoch_handlers = post_epoch_handlers
        self.silent = silent

    def train(self, model: Model, data_loader: DataLoader, params: TrainingParams) -> None:
        """
        Performs the training loop for Model instance. All type of handlers are involved,
        however the epoch training logic is encapsulated
        via @abstractmethod to give the space for extensibility.
        :param model: the Model instance to be trained
        :param data_loader: entity which provides the model with training data and training labels
        :param params: the set of cycle's parameters
        :return: None
        """
        for _ in range(params.epoch_count):
            for handler in self.pre_epoch_handlers:
                handler.handle(model, mode=HandlerMode.PRE_EPOCH)

            self.train_epoch(model, data_loader, params, self.after_step_handlers)

            for handler in self.post_epoch_handlers:
                handler.handle(model, mode=HandlerMode.POST_EPOCH)
            model.learning_info.epochs_trained += 1

    @abstractmethod
    def train_epoch(self, model: Model, data_loader: DataLoader, params: TrainingParams,
                    after_step_handlers: List[LearningHandler] = None) -> None:
        """
        Trains the Model instance for one epoch with data provided.
        :param model: the Model instance to be trained
        :param data_loader: entity which provides the model with training data and training labels
        :param params: the set of cycle's parameters
        :param after_step_handlers: set of handlers to invoke after a step of learning is performed.
        :return: None.
        """


class DefaultTrainer(Trainer):
    """
    Implementation of Trainer which simply iterates over the batches_count
    of batches in a train_epoch method.
    """

    def train_epoch(self, model: Model, data_loader: DataLoader, params: TrainingParams,
                    after_step_handlers: List[LearningHandler] = None) -> None:
        """Iterates over params.batch_count of batches, to train on them.
        Invokes after_step_handlers after pushing the gradients backward."""
        for _ in range(params.batch_count):
            data_batch, labels_batch = data_loader.get_batch()
            model_output = model(data_batch)
            loss = model.loss_function(model_output, labels_batch)
            model.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            model.optimizer.step()
            # model.learning_info.last_loss = loss.item()
            for handler in after_step_handlers:
                handler.handle(model, mode=HandlerMode.STEP)

from tqdm import tqdm

from src.models import Model
from src.trainers.handlers import LearningHandler, HandlerMode


class Printer:
    """Provides nice console output for the learning process. Uses tqdm progressbars."""

    def __init__(self, epochs: int, batches_num: int):
        self.batches_num = batches_num
        self._epochs_progressbar = tqdm(total=epochs, leave=False)
        self._step_progressbar = tqdm(total=self.batches_num, leave=False)

    def update_step(self):
        """Updates step progressbar"""
        self._step_progressbar.update()

    def update_epoch(self):
        """Updates epoch progressbar and resets step progressbar"""
        self._epochs_progressbar.update()
        self._step_progressbar.reset(total=self.batches_num)

    def update_accuracy(self, val_accuracy: float, train_accuracy: float):
        """Prints model's accuracy values"""
        self._epochs_progressbar.set_postfix_str(
            "val_accuracy: {:.6f}, train_accuracy: {:.6f}".format(val_accuracy, train_accuracy))


class PrinterHandler(LearningHandler):
    """Printer Wrapper used in a training cycle.
    Performs console output using model's learning_info."""

    def __init__(self, printer: Printer):
        self._printer = printer
        self._printer.update_accuracy(0.0, 0.0)

    def handle(self, model: Model, mode: HandlerMode = HandlerMode.NONE) -> None:
        if mode == HandlerMode.POST_EPOCH:
            self._printer.update_epoch()
            if len(model.learning_info.val_accuracy_history) > 0 and len(
                    model.learning_info.train_accuracy_history) > 0:
                self._printer.update_accuracy(model.learning_info.val_accuracy_history[-1],
                                              model.learning_info.train_accuracy_history[-1])
        elif mode == HandlerMode.STEP:
            self._printer.update_step()
        else:
            raise ValueError(f"Unknown HandlerMode {mode.name}.")

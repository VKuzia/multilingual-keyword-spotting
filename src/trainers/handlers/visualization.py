from tqdm import tqdm

from src.models import Model
from src.trainers.handlers import LearningHandler, HandlerMode


class Printer:

    def __init__(self, epochs: int, batches_num: int):
        self.batches_num = batches_num
        self.epochs_progressbar = tqdm(total=epochs, leave=False)
        self.step_progressbar = tqdm(total=self.batches_num, leave=False)

    def update_step(self):
        self.step_progressbar.update()
        self.step_progressbar.set_description()

    def update_epoch(self):
        self.epochs_progressbar.update()
        self.step_progressbar.reset(total=self.batches_num)

    def update_loss(self, loss: float):
        self.step_progressbar.set_postfix_str("Loss: {:.4f}".format(loss))

    def update_accuracy(self, val_accuracy: float, train_accuracy: float):
        self.epochs_progressbar.set_postfix_str(
            "val_accuracy: {:.6f}, train_accuracy: {:.6f}".format(val_accuracy, train_accuracy))


class PrinterHandler(LearningHandler):

    def __init__(self, printer: Printer):
        self.printer = printer
        self.printer.update_accuracy(0.0, 0.0)

    def handle(self, model: Model, mode: HandlerMode = HandlerMode.NONE) -> None:
        if mode == HandlerMode.POST_EPOCH:
            self.printer.update_epoch()
            self.printer.update_accuracy(model.learning_info.val_accuracy,
                                         model.learning_info.train_accuracy)
        elif mode == HandlerMode.STEP:
            self.printer.update_step()
            self.printer.update_loss(model.learning_info.last_loss)
        else:
            raise ValueError(f"Unknown HandlerMode {mode.name}.")

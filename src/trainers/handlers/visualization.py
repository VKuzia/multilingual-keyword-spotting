from typing import List, Dict

from tqdm import tqdm

from src.models import Model
from src.trainers.handlers import LearningHandler, HandlerMode


class Printer:
    """Provides nice console output for the learning process. Uses tqdm progressbars."""

    def __init__(self, epochs: int, batches_num: int, metrics: List[str]):
        self.batches_num = batches_num
        self.metrics = sorted(metrics)
        self._epochs_progressbar = tqdm(total=epochs, leave=False)
        self._step_progressbar = tqdm(total=self.batches_num, leave=False)

    def update_step(self):
        """Updates step progressbar"""
        self._step_progressbar.update()

    def update_epoch(self):
        """Updates epoch progressbar and resets step progressbar"""
        self._epochs_progressbar.update()
        self._step_progressbar.reset(total=self.batches_num)

    def update_metrics(self, metrics_dict: Dict[str, float]):
        """Prints model's metrics to epoch progressbar.
        Uses keys from intersection of self.metrics and given dict"""
        content = '; '.join(['{}({:.3f})'.format(key, metrics_dict[key]) for key in self.metrics if
                             key in metrics_dict.keys()])
        self._epochs_progressbar.set_postfix_str(content)


class PrinterHandler(LearningHandler):
    """Printer Wrapper used in a training cycle.
    Performs console output using model's learning_info."""

    def __init__(self, printer: Printer):
        self._printer = printer
        self._printer.update_metrics({})

    def handle(self, model: Model, mode: HandlerMode = HandlerMode.NONE) -> None:
        if mode == HandlerMode.POST_EPOCH:
            self._printer.update_epoch()
            metrics_dict = {}
            for key, value in model.learning_info.metrics.items():
                if value:
                    metrics_dict[key] = value[-1]
            self._printer.update_metrics(metrics_dict)
        elif mode == HandlerMode.STEP:
            self._printer.update_step()
        else:
            raise ValueError(f"Unknown HandlerMode {mode.name}.")

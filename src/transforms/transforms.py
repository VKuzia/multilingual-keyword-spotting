import torch
from torch import Tensor

from src.utils.helpers import happen


class TimeShifter:
    def __init__(self, shift: int, forward_probability: float = 0.5):
        self.forward_probability = forward_probability
        self.shift = shift

    def __call__(self, spectrogram: Tensor) -> torch.Tensor:
        if happen(self.forward_probability):
            return torch.roll(spectrogram,
                              shifts=(self.shift,),
                              dims=(len(spectrogram.shape) - 1))
        else:
            return torch.roll(spectrogram,
                              shifts=(-self.shift,),
                              dims=(len(spectrogram.shape) - 1))

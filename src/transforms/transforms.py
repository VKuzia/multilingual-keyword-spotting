import torch
from torch import Tensor

from src.utils import happen


class TimeShifter:
    """
    Provides time shift transform on given Tensor (aka MelSpectrogram).
    Uses torch.roll, which performs a cyclic (!) shift, so it's useful only for small shifts
    """

    def __init__(self, shift: int):
        self.shift = shift

    def __call__(self, spectrogram: Tensor) -> torch.Tensor:
        return torch.roll(spectrogram, shifts=(0, self.shift), dims=(0, 1))


class PowerEnhance:
    """
    Provides time shift transform on given Tensor (aka MelSpectrogram).
    Uses torch.roll, which performs a cyclic (!) shift, so it's useful only for small shifts
    """

    def __init__(self, power: float):
        self.power = power

    def __call__(self, spectrogram: Tensor) -> torch.Tensor:
        return torch.pow(spectrogram, self.power)

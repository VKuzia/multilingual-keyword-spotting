from abc import abstractmethod, ABC
from typing import List, Tuple, Callable

import torch
import torchaudio.transforms as T
from torch import Tensor

from src.transforms.transforms import TimeShifter
from src.utils.helpers import happen


class Transformer(ABC):
    """Provides MelSpectrogram conversion and random augmentation functionality"""

    @abstractmethod
    def to_mel_spectrogram(self, waveform: Tensor) -> Tensor:
        """Converts given waveform into a MelSpectrogram"""
        pass

    @property
    @abstractmethod
    def augment_list(self) -> List[Tuple[float, Callable]]:
        """List of augmentations and their probability.
        Augmentations are either user-defined callable functions or
        PyTorch transforms"""
        pass

    @property
    @abstractmethod
    def shape(self) -> Tuple[int, int]:
        """Default shape of output MelSpectrogram"""
        pass

    def augment(self, spectrogram: Tensor) -> Tensor:
        """Iterates over self.augment_list
        and applies transformations with corresponding probabilities"""
        result = spectrogram
        for probability, transform in self.augment_list:
            if happen(probability):
                result = transform(result)
        return result

    def pad_to_shape(self, spectrogram: Tensor, value: float = 0.0) -> Tensor:
        """Aligns last dim of given spectrogram to fit into default shape"""
        if spectrogram.shape == self.shape:
            return spectrogram
        target = torch.full(size=self.shape, fill_value=value)
        target[..., :spectrogram.shape[-1]] = spectrogram[...,
                                              :min(spectrogram.shape[-1], self.shape[-1])]
        return target


class DefaultTransformer(Transformer):

    def __init__(self):
        self._to_mel_spectrogram = T.MelSpectrogram(n_mels=self.shape[0],
                                                    hop_length=401,
                                                    power=0.75)
        frequency_masking = torch.nn.Sequential(T.FrequencyMasking(2),
                                                T.FrequencyMasking(2))
        time_masking = torch.nn.Sequential(T.TimeMasking(2),
                                           T.TimeMasking(2))
        # time_stretch_1 = T.TimeStretch(n_freq=self.shape[0], fixed_rate=0.95)
        # time_stretch_2 = T.TimeStretch(n_freq=self.shape[0], fixed_rate=1.05)
        # time_stretch_and_frequency = torch.nn.Sequential(time_stretch_1,
        #                                                  frequency_masking)
        time_shifter = TimeShifter(3)
        self._augmentations = [
            (0.25, time_shifter),
            (0.25, frequency_masking),
            (0.25, time_masking),
            # (0.1, time_stretch_1),
            # (0.1, time_stretch_2),
            # (0.1, time_stretch_and_frequency),
        ]

    @property
    def shape(self) -> Tuple[int, int]:
        return 49, 40

    def to_mel_spectrogram(self, waveform: Tensor) -> Tensor:
        return self.pad_to_shape(self._to_mel_spectrogram(waveform))

    @property
    def augment_list(self) -> List[Tuple[float, Callable]]:
        return self._augmentations


class ValidationTransformer(Transformer):

    def __init__(self):
        self._to_mel_spectrogram = T.MelSpectrogram(n_mels=self.shape[0],
                                                    hop_length=401,
                                                    power=0.75)

    def to_mel_spectrogram(self, waveform: Tensor) -> Tensor:
        return self.pad_to_shape(self._to_mel_spectrogram(waveform))

    @property
    def augment_list(self) -> List[Tuple[float, Callable]]:
        return []

    @property
    def shape(self) -> Tuple[int, int]:
        return 49, 40
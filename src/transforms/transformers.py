from abc import abstractmethod, ABC
from typing import List, Tuple, Callable

import torch
import torchaudio.transforms as T
from torch import Tensor

from src.transforms.transforms import TimeShifter, PowerEnhance
from src.utils import happen


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
    """
    A slight SpecAugment transformer.
    Applies frequency masking, time masking and time shifts
    """

    def __init__(self):
        self._to_mel_spectrogram = T.MelSpectrogram(n_mels=self.shape[0],
                                                    hop_length=401,
                                                    power=0.75)
        frequency_masking = torch.nn.Sequential(T.FrequencyMasking(2),
                                                T.FrequencyMasking(2))
        time_masking = torch.nn.Sequential(T.TimeMasking(2),
                                           T.TimeMasking(2))
        time_shifter = TimeShifter(3)
        self._augmentations = [
            (0.25, time_shifter),
            (0.25, frequency_masking),
            (0.25, time_masking),
        ]

    @property
    def shape(self) -> Tuple[int, int]:
        return 49, 40

    def to_mel_spectrogram(self, waveform: Tensor) -> Tensor:
        return self.pad_to_shape(self._to_mel_spectrogram(waveform))

    @property
    def augment_list(self) -> List[Tuple[float, Callable]]:
        return self._augmentations


class SpecAugTransformer(Transformer):
    """
    A slight SpecAugment transformer.
    Applies frequency masking, time masking and time shifts
    """

    def __init__(self):
        self._to_mel_spectrogram = T.MelSpectrogram(n_mels=self.shape[0],
                                                    hop_length=401)
        frequency_masking_1 = torch.nn.Sequential(T.FrequencyMasking(2),
                                                  T.FrequencyMasking(2))
        frequency_masking_2 = T.FrequencyMasking(3)
        time_masking = torch.nn.Sequential(T.TimeMasking(2),
                                           T.TimeMasking(2))
        time_shifter_1 = TimeShifter(3)
        time_shifter_2 = TimeShifter(-4)

        power_enhancer_1 = PowerEnhance(1.07)

        power_enhancer_2 = PowerEnhance(0.9)
        self._augmentations = [
            (0.4, frequency_masking_1),
            (0.3, frequency_masking_2),
            (0.4, time_masking),
            (0.3, time_shifter_1),
            (0.3, time_shifter_2),
            (0.2, power_enhancer_1),
            (0.2, power_enhancer_2)
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
    """
    Identity transformator. Is used in a validation loop for now.
    TODO: apply augmentations to few-shot targets
    """

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

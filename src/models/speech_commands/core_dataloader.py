import os
from abc import ABC, abstractmethod
from enum import Enum
from typing import Tuple, List, Callable

import torch
import torchaudio

from torchaudio.datasets import SPEECHCOMMANDS
from src.dataloaders.dataloader import DataLoader


class SpeechCommandsMode(Enum):
    """Specifies the part of dataset to use in loader."""
    TRAINING = 0
    VALIDATION = 1
    TESTING = 2


class SpeechCommandsDataset(SPEECHCOMMANDS):
    """
    Implements a wrapper of SpeechCommands dataset.
    Kindly taken from
    https://pytorch.org/tutorials/intermediate/speech_command_recognition_with_torchaudio.html
    """

    def __init__(self, path: str, subset: SpeechCommandsMode,
                 predicate: Callable[[str], bool] = lambda label: True):
        super().__init__(path, download=False, subset=str(subset.name).lower())

        if subset == SpeechCommandsMode.VALIDATION:
            self._walker = self.load_list("validation_list.txt", predicate)
        elif subset == SpeechCommandsMode.TESTING:
            self._walker = self.load_list("testing_list.txt", predicate)
        elif subset == SpeechCommandsMode.TRAINING:
            excludes = self.load_list("validation_list.txt") + self.load_list("testing_list.txt")
            excludes = set(excludes)
            self._walker = [w for w in self._walker if
                            w not in excludes and predicate(w.split('/')[-2])]

    def load_list(self, filename: str,
                  predicate: Callable[[str], bool] = lambda label: True) -> List[str]:
        """Reads specified Speech Commands file to choose samples to provide from dataset"""
        filepath = os.path.join(self._path, filename)
        with open(filepath) as file:
            result = []
            for line in file:
                if predicate(self.line_to_label(line)):
                    result.append(os.path.join(self._path, line.strip()))
            return result

    @staticmethod
    def line_to_label(line: str) -> str:
        return line.split('/')[0]


class SpeechCommandsBase(DataLoader, ABC):
    """
    Provides basic functionality for all Speech Commands related wrappers
    """

    labels: List[str] = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go",
                         "zero", "one", "two",
                         "three", "four", "five", "six", "seven", "eight", "nine", "unknown"]

    transform = torchaudio.transforms. \
        Spectrogram(n_fft=97, hop_length=400)  # creates spectrograms of 1x49x40

    @staticmethod
    def pad_sequence(batch):
        """Make all tensor in a batch the same length by padding with zeros"""
        batch = [item.t() for item in batch]
        batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.)
        return batch.permute(0, 2, 1)

    def collate_fn(self, batch):
        """Creates a batch of samples"""
        tensors, targets = [], []

        # initial data tuple has the form: waveform, sample_rate, label, speaker_id, utter_number
        for waveform, _, label, *_ in batch:
            tensors += [waveform]
            targets += [torch.tensor(self.label_to_index(label))]

        # Group the list of tensors into a batched tensor
        tensors = self.pad_sequence(tensors)
        targets = torch.stack(targets)

        return tensors, targets

    @abstractmethod
    def label_to_index(self, word: str) -> int:
        """Maps str keyword into integer label for models to be fit on"""


class CoreDataLoader(SpeechCommandsBase):
    """
    Implements DataLoader interface for SpeechCommands dataset.
    Mainly taken from
    https://pytorch.org/tutorials/intermediate/speech_command_recognition_with_torchaudio.html
    """

    def __init__(self, path: str, mode: SpeechCommandsMode, batch_size: int, cuda: bool = True):
        super().__init__()
        self.dataset = SpeechCommandsDataset(path, mode)
        self.batch_size = batch_size
        self.loader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
            pin_memory=cuda
        )
        self.cuda = cuda

    def get_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return next(self._yield_batch())

    def _yield_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Takes next batch of the self.loader, transforms it and yields."""
        for data, labels in self.loader:
            transformed_data = self.transform(data).repeat(1, 3, 1, 1)
            if self.cuda:
                yield transformed_data.to('cuda'), labels.to('cuda')
            else:
                yield transformed_data, labels

    def label_to_index(self, word: str) -> int:
        """Returns the index of the given word"""
        if word in self.labels:
            return self.labels.index(word)
        else:
            return len(self.labels) - 1  # unknown label

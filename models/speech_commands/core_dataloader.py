import os
from enum import Enum
from typing import Tuple, List

import torch
import torchaudio

from dataloaders.dataloader import DataLoader
from torchaudio.datasets import SPEECHCOMMANDS


class SpeechCommandsDataset(SPEECHCOMMANDS):
    """
    Implements a wrapper of SpeechCommands dataset.
    Kindly taken from https://pytorch.org/tutorials/intermediate/speech_command_recognition_with_torchaudio.html
    """

    def __init__(self, path: str, subset: str):
        super().__init__(path, download=False)

        def load_list(filename):
            filepath = os.path.join(self._path, filename)
            with open(filepath) as file:
                return [os.path.join(self._path, line.strip()) for line in file]

        if subset == "validation":
            self._walker = load_list("validation_list.txt")
        elif subset == "testing":
            self._walker = load_list("testing_list.txt")
        elif subset == "training":
            excludes = load_list("validation_list.txt") + load_list("testing_list.txt")
            excludes = set(excludes)
            self._walker = [w for w in self._walker if w not in excludes]


class CoreDataLoader(DataLoader):
    """
    Implements DataLoader interface for SpeechCommands dataset.
    Mainly taken from https://pytorch.org/tutorials/intermediate/speech_command_recognition_with_torchaudio.html

    TODO: fix typing.
    """

    labels: List[str] = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go", "zero", "one", "two",
                         "three", "four", "five", "six", "seven", "eight", "nine", "unknown"]

    class Mode(Enum):
        """Specifies the part of dataset to use in loader."""
        TRAINING = 0,
        VALIDATION = 1,
        TESTING = 2,

    def __init__(self, path: str, mode: Mode, batch_size: int):
        """TODO: specify, whether to use cuda or not."""
        self.dataset = SpeechCommandsDataset(path, str(mode.name).lower())
        self.batch_size = batch_size
        self.transform = torchaudio.transforms.Spectrogram(n_fft=97, hop_length=400)  # creates spectrograms of 1x49x40
        self.loader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
            pin_memory=True
        )

    def get_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return next(self._yield_batch())

    def _yield_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Takes next batch of the self.loader, transforms it and yields."""
        for data, labels in self.loader:
            yield self.transform(data).repeat(1, 3, 1, 1).to('cuda'), labels.to('cuda')

    def label_to_index(self, word):
        """Returns the index of the given word"""
        if word in self.labels:
            return torch.tensor(self.labels.index(word))
        else:
            return torch.tensor(len(self.labels) - 1)  # unknown

    @staticmethod
    def pad_sequence(batch):
        """Make all tensor in a batch the same length by padding with zeros"""
        batch = [item.t() for item in batch]
        batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.)
        return batch.permute(0, 2, 1)

    def collate_fn(self, batch):
        """Creates a batch of samples"""
        tensors, targets = [], []

        # initial data tuple has the form: waveform, sample_rate, label, speaker_id, utterance_number
        for waveform, _, label, *_ in batch:
            tensors += [waveform]
            targets += [self.label_to_index(label)]

        # Group the list of tensors into a batched tensor
        tensors = self.pad_sequence(tensors)
        targets = torch.stack(targets)

        return tensors, targets

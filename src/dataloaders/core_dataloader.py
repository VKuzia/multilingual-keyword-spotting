from abc import ABC, abstractmethod
from typing import Tuple

import torch
import torchaudio

from src.dataloaders import DataLoader
from src.dataloaders.dataset import WalkerDataset


class BaseDataLoader(DataLoader, ABC):
    # TODO: Improve use of CUDA

    transform = torchaudio.transforms. \
        Spectrogram(n_fft=97, hop_length=400)  # creates spectrograms of 1x49x40

    @abstractmethod
    def __init__(self, dataset: WalkerDataset, batch_size: int, cuda: bool = True):
        self.dataset = dataset
        self.labels = self.dataset.labels
        self.batch_size = batch_size
        self.cuda = cuda

    @staticmethod
    def pad_sequence(batch):
        """Make all tensor in a batch the same length by padding with zeros"""
        batch = [item.t() for item in batch]
        batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.)
        return batch.permute(0, 2, 1)

    def collate_fn(self, batch):
        """Creates a batch of samples"""
        tensors, targets = [], []

        # initial data tuple has the form: waveform, sample_rate, label
        for waveform, _, label in batch:
            tensors += [waveform]
            targets += [torch.tensor(self.label_to_index(label))]

        # Group the list of tensors into a batched tensor
        tensors = self.pad_sequence(tensors)
        targets = torch.stack(targets)

        return tensors, targets

    @abstractmethod
    def label_to_index(self, word: str) -> int:
        """Maps str keyword into integer label for models to be fit on"""


class ClassificationDataLoader(BaseDataLoader):

    def __init__(self, dataset: WalkerDataset, batch_size: int, cuda: bool = True):
        super().__init__(dataset, batch_size, cuda)
        self.loader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
            pin_memory=cuda
        )

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
            if self.dataset.unknown_index:
                return self.dataset.unknown_index
            else:
                raise ValueError("Unknown label but 'unknown' is not a target category")

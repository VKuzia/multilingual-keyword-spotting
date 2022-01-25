from abc import ABC, abstractmethod
from typing import Tuple

import torch
import torchaudio

from src.dataloaders import DataLoader
from src.dataloaders.base import WalkerDataset


class BaseDataLoader(DataLoader, ABC):
    # TODO: Improve use of CUDA

    transform = torchaudio.transforms. \
        Spectrogram(n_fft=97, hop_length=400)  # creates spectrograms of 1x49x40

    @abstractmethod
    def __init__(self, batch_size: int, cuda: bool = True):
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
        self.dataset = dataset
        self.labels = self.dataset.labels
        super().__init__(batch_size, cuda)
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


class FewShotDataLoader(BaseDataLoader):

    def __init__(self, target_dataset: WalkerDataset,
                 non_target_dataset: WalkerDataset, batch_size: int,
                 target: str, target_probability: float, cuda: bool = True):
        super().__init__(batch_size, cuda)
        self.target = target
        self.target_dataset = target_dataset
        self.non_target_dataset = non_target_dataset
        target_batch_size = int(batch_size * target_probability)
        non_target_batch_size = batch_size - target_batch_size
        self._target_loader = torch.utils.data.DataLoader(
            self.target_dataset,
            batch_size=target_batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
            pin_memory=cuda
        )
        self._non_target_loader = torch.utils.data.DataLoader(
            self.non_target_dataset,
            batch_size=non_target_batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
            pin_memory=cuda
        )

    def label_to_index(self, word: str) -> int:
        return 1 if word == self.target else 0

    def get_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return next(self._yield_batch())

    def _yield_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        for (target_data, target_labels), (non_target_data, non_target_labels) in zip(
                self._target_loader,
                self._non_target_loader):
            data = torch.concat((target_data, non_target_data), dim=0)
            labels = torch.concat((target_labels, non_target_labels), dim=0)
            transformed_data = self.transform(data).repeat(1, 3, 1, 1)
            if self.cuda:
                yield transformed_data.to('cuda'), labels.to('cuda')
            else:
                yield transformed_data, labels

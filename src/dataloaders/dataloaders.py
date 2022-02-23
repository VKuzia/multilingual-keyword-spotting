from abc import ABC, abstractmethod
from typing import Tuple, List

import torch
import torchaudio
from torch import Tensor

from src.dataloaders import DataLoader, Dataset


class BaseDataLoader(DataLoader, ABC):
    # TODO: Generalize transform usage

    transform = torchaudio.transforms. \
        Spectrogram(n_fft=97, hop_length=400)  # creates spectrograms of 1x49x40

    @abstractmethod
    def __init__(self, batch_size: int, cuda: bool = True):
        self.batch_size = batch_size
        self.device = torch.device('cuda' if cuda else 'cpu')

    @staticmethod
    def pad_sequence(batch) -> Tensor:
        """Make all tensor in a batch the same length by padding with zeros"""
        batch = [item.t() for item in batch]
        batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.)
        return batch.permute(0, 2, 1)

    def collate_fn(self, batch) -> Tuple[Tensor, Tensor]:
        """Creates a batch of samples"""
        tensors, targets = [], []

        # initial data tuple has the form: waveform, sample_rate, label
        for waveform, _, label in batch:
            tensors += [waveform]
            targets += [torch.tensor(self.label_to_index(label), device=self.device)]

        # Group the list of tensors into a batched tensor
        tensors = self.pad_sequence(tensors)
        targets = torch.stack(targets)

        return tensors, targets

    @abstractmethod
    def label_to_index(self, word: str) -> int:
        """Maps str keyword into integer label for models to be fit on"""


class ClassificationDataLoader(BaseDataLoader):
    """Implements batch loading for given Dataset instance. Each word has it's own label category"""

    def __init__(self, dataset: Dataset, batch_size: int, cuda: bool = True, workers: int = 0):
        self._dataset = dataset
        self.labels = self._dataset.labels
        self.labels_map = {word: index for index, word in enumerate(self.labels)}
        super().__init__(batch_size, cuda)
        self._loader = torch.utils.data.DataLoader(
            self._dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
            pin_memory=cuda,
            num_workers=workers
        )
        self._loader_iter = iter(self._loader)

    def get_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        try:
            data, labels = next(self._loader_iter)
        except StopIteration:
            self._loader_iter = iter(self._loader)
            data, labels = next(self._loader_iter)
        return self.transform(data).to(self.device, non_blocking=True), labels.to(self.device,
                                                                                  non_blocking=True)

    def get_labels(self) -> List[str]:
        return self._dataset.labels

    def label_to_index(self, word: str) -> int:
        """Returns the index of the given word"""
        if word in self.labels_map.keys():
            return self.labels_map[word]
        else:
            if self._dataset.unknown_index:
                return self._dataset.unknown_index
            else:
                raise ValueError("Unknown label but 'unknown' is not a target category")


class FewShotDataLoader(BaseDataLoader):
    """Implements batch loading for given target and non_target Dataset instances.
    Each word is classified either as unknown, or as target"""

    def get_labels(self) -> List[str]:
        return ['unknown', 'target']

    def __init__(self, target_dataset: Dataset,
                 non_target_dataset: Dataset, batch_size: int,
                 target: str, target_probability: float, cuda: bool = True, workers: int = 0):
        super().__init__(batch_size, cuda)
        self.target = target
        self._target_dataset = target_dataset
        self._non_target_dataset = non_target_dataset
        target_batch_size = int(batch_size * target_probability)
        non_target_batch_size = batch_size - target_batch_size
        self._target_loader = torch.utils.data.DataLoader(
            self._target_dataset,
            batch_size=target_batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
            pin_memory=cuda,
            num_workers=workers
        )
        self._non_target_loader = torch.utils.data.DataLoader(
            self._non_target_dataset,
            batch_size=non_target_batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
            pin_memory=cuda,
            num_workers=workers
        )
        self._target_loader_iter = iter(self._target_loader)
        self._non_target_loader_iter = iter(self._non_target_loader)

    def label_to_index(self, word: str) -> int:
        return 1 if word == self.target else 0

    def get_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:

        try:
            target_data, target_labels = next(self._target_loader_iter)
        except StopIteration:
            self._target_loader_iter = iter(self._target_loader)
            target_data, target_labels = next(self._target_loader_iter)
        try:
            non_target_data, non_target_labels = next(self._non_target_loader_iter)
        except StopIteration:
            self._non_target_loader_iter = iter(self._non_target_loader)
            non_target_data, non_target_labels = next(self._non_target_loader_iter)

        data = torch.concat((target_data, non_target_data), dim=0)
        labels = torch.concat((target_labels, non_target_labels), dim=0)
        return data.to(self.device, non_blocking=True), labels.to(self.device, non_blocking=True)

from abc import ABC, abstractmethod
from typing import Tuple, List

import torch
from torch import Tensor

from src.dataloaders import DataLoader, Dataset


class BaseDataLoader(DataLoader, ABC):

    @abstractmethod
    def __init__(self, batch_size: int, cuda: bool = True):
        self.batch_size = batch_size
        self.device = torch.device('cuda' if cuda else 'cpu')

    def _label_to_index(self, label: str) -> Tensor:
        return torch.tensor(self.label_to_index(label), device=self.device)

    @staticmethod
    def pad_sequence(batch) -> Tensor:
        """Make all tensor in a batch the same length by padding with zeros"""
        batch = [item.t() for item in batch]
        batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.)
        return batch

    def collate_fn(self, batch) -> Tuple[Tensor, Tensor]:
        """Creates a batch of samples"""
        tensors, targets = [], []

        # initial data tuple has the form: spectrogram, label
        for tensor, label in batch:
            tensors += [tensor]
            targets += [torch.tensor(self.label_to_index(label), device=self.device)]

        return self.pad_sequence(tensors), torch.stack(targets)

    @abstractmethod
    def label_to_index(self, word: str) -> int:
        """Maps str keyword into integer label for models to be fit on"""


class ClassificationDataLoader(BaseDataLoader):
    """Implements batch loading for given Dataset instance. Each word has it's own label category"""

    def get_batch_count(self) -> int:
        return (len(self._dataset) + self.batch_size - 1) // self.batch_size

    def __init__(self, dataset: Dataset, batch_size: int, cuda: bool = True):
        self._dataset = dataset
        self.labels = self._dataset.labels
        self.labels_map = {word: index for index, word in enumerate(self.labels)}
        super().__init__(batch_size, cuda)
        self._loader = torch.utils.data.DataLoader(
            self._dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn
            # pin_memory=cuda
        )
        self._loader_iter = iter(self._loader)

    def get_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        try:
            data, labels = next(self._loader_iter)
        except StopIteration:
            self.reset()
            data, labels = next(self._loader_iter)
        if data.shape[1] != 1:
            data = torch.unsqueeze(data, 1)
        return data.to(self.device, non_blocking=True), labels.to(self.device, non_blocking=True)

    def get_labels(self) -> List[str]:
        return self._dataset.labels

    def label_to_index(self, word: str) -> int:
        """Returns the index of the given word, handles unknown label specifically"""
        if word in self.labels_map.keys():
            return self.labels_map[word]
        else:
            if self._dataset.unknown_index:
                return self._dataset.unknown_index
            else:
                raise ValueError("Unknown label but 'unknown' is not a target category")

    def reset(self):
        self._loader_iter = iter(self._loader)

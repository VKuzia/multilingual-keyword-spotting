import os
from abc import ABC, abstractmethod
from enum import Enum
from typing import Tuple, List, Optional

import torch
import torchaudio
from torch import Tensor


class DataLoaderMode(Enum):
    """Specifies the part of dataset to use in loader."""
    TRAINING = 0
    VALIDATION = 1
    TESTING = 2


class DataLoader(ABC):
    """
    Base class for every data loading entity in the project.
    Must provide get_batch function to be used while learning.
    """

    @abstractmethod
    def get_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Produces a portion (batch) of data to feed into the Model instance.
        :return: tuple of (X, y), where X is the input data batch and y is the labels batch
        """

    @abstractmethod
    def get_labels(self) -> List[str]:
        pass


class Dataset(ABC):

    @abstractmethod
    def __getitem__(self, n: int) -> Tuple[Tensor, int, str]:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

    @property
    @abstractmethod
    def unknown_index(self) -> Optional[int]:
        pass

    @property
    @abstractmethod
    def labels(self) -> List[str]:
        pass

class WalkerDataset(Dataset, ABC):

    @property
    @abstractmethod
    def train_list(self) -> str:
        pass

    @property
    @abstractmethod
    def validation_list(self) -> str:
        pass

    @property
    @abstractmethod
    def test_list(self) -> str:
        pass

    @property
    @abstractmethod
    def path_to_clips(self) -> str:
        pass

    @abstractmethod
    def extract_label_short(self, path_to_clip: str) -> str:
        pass

    @abstractmethod
    def extract_label_full(self, path_to_clip: str) -> str:
        pass

    @abstractmethod
    def __init__(self, root: str, subset: DataLoaderMode,
                 predicate=lambda label: True):
        self.root = root
        self.subset = subset
        # TODO: VALIDATION handling.
        if subset == DataLoaderMode.TRAINING:
            self._walker = self.load_list(self.train_list, predicate)
        elif subset == DataLoaderMode.TESTING:
            self._walker = self.load_list(self.test_list, predicate)
        elif subset == DataLoaderMode.VALIDATION:
            self._walker = self.load_list(self.validation_list, predicate)
        else:
            raise ValueError(f"Can't handle unknown DataLoaderMode '{subset.name}'")

    def load_list(self, filename: str,
                  predicate=lambda label: True) -> List[str]:
        """Reads specified file to choose samples to provide from dataset"""
        filepath = os.path.join(self.root, filename)
        with open(filepath) as file:
            result = []
            for line in file:
                if predicate(self.extract_label_short(line)):
                    result.append(os.path.join(self.root + self.path_to_clips, line.strip()))
            return result

    def __getitem__(self, n: int) -> Tuple[Tensor, int, str]:
        label = self.extract_label_full(self._walker[n])
        waveform, sample_rate = torchaudio.load(self._walker[n])
        return waveform, sample_rate, label

    def __len__(self) -> int:
        return len(self._walker)


class MultiWalkerDataset(Dataset):

    @property
    def unknown_index(self) -> Optional[int]:
        return None

    @property
    def labels(self) -> List[str]:
        return self._labels

    def __init__(self, datasets: List[Dataset]):
        if not datasets:
            raise ValueError('Empty datasets creation is not allowed.')
        self._datasets = datasets
        self._lengths = [len(dataset) for dataset in datasets]
        self._len = sum(self._lengths)
        self._labels = sum([dataset.labels for dataset in datasets], [])

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, n: int) -> Tuple[Tensor, int, str]:
        if n >= self._len or n < 0:
            raise IndexError(f'Could not produce item of index {n}')
        index: int = 0
        aggregated: int = 0
        while index < len(self._lengths) and n - aggregated >= self._lengths[index]:
            aggregated += self._lengths[index]
            index += 1
        return self._datasets[index][n - aggregated]

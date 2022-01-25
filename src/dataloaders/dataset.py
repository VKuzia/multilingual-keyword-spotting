import os
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import List, Tuple, Optional

import torchaudio
from torch import Tensor

from src.dataloaders import DataLoaderMode


class WalkerDataset(ABC):

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

    @property
    @abstractmethod
    def labels(self) -> List[str]:
        pass

    @property
    @abstractmethod
    def unknown_index(self) -> Optional[int]:
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

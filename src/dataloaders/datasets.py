import os
from abc import ABC, abstractmethod
from enum import Enum
from typing import Tuple, List, Optional, Callable

import pandas as pd
import torch
import torchaudio
from torch import Tensor

from src.transforms import Transformer
from src.utils import happen


class DataLoaderMode(Enum):
    """Specifies the part of dataset to use in loader."""
    TRAINING = 0
    VALIDATION = 1
    TESTING = 2
    ANY = 4


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
    def get_batch_count(self) -> int:
        pass

    @abstractmethod
    def get_labels(self) -> List[str]:
        pass

    @abstractmethod
    def reset(self):
        pass


class Dataset(ABC):
    """
    Provides basic methods of pytorch-used datasets.
    """

    def __init__(self, is_wav: bool):
        self._is_wav = is_wav

    @abstractmethod
    def __getitem__(self, n: int) -> Tuple[Tensor, str]:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

    @property
    def is_wav(self) -> bool:
        """Shows whether dataset loads ready-made spectrograms
        or provides waveforms"""
        return self._is_wav

    @property
    @abstractmethod
    def unknown_index(self) -> Optional[int]:
        """Returns index of 'unknown' label for given dataset.
         If there should be that label, returns None"""
        pass

    @property
    @abstractmethod
    def labels(self) -> List[str]:
        """List of all target labels of the dataset"""
        pass


def is_word_predicate(word: str) -> Callable[[str], bool]:
    """Predicate to check whether given short path to audio targets word provided"""
    return lambda x: x.split('/')[0] == word


class TableDataset(Dataset):
    """
    Reads given csv and provides data from root dataset's dir. Accepts filtering by mode or lambda.
    """

    @property
    def unknown_index(self) -> Optional[int]:
        return None

    def __init__(self, root: str, data_csv: str, subset: DataLoaderMode, is_wav: bool = True,
                 predicate=lambda label: True):
        super().__init__(is_wav)
        self.root = root
        self.subset = subset
        if subset == DataLoaderMode.TRAINING:
            mode = 'train'
        elif subset == DataLoaderMode.TESTING:
            mode = 'test'
        elif subset == DataLoaderMode.VALIDATION:
            mode = 'val'
        elif subset == DataLoaderMode.ANY:
            mode = 'any'
        else:
            raise ValueError(f"Can't handle unknown DataLoaderMode '{subset.name}'")
        self.data = self._load_by_mode(data_csv, mode, predicate)
        self._load_method = self._load_wav if self.is_wav else self._load_spec
        self._labels = sorted(list(self.data['label'].unique()))
        if not self.is_wav:
            self.data['path'] = self.data['path'].apply(lambda x: x.replace(".wav", ".pt"))

    def _load_by_mode(self, data_csv: str, mode: str, predicate=lambda label: True) -> pd.DataFrame:
        """Reads given csv and filters it according given mode
        returning resulting subset DataFrame"""
        data = pd.read_csv(data_csv, delimiter=',')
        if mode != 'any':
            subset = data[data['mode'] == mode]
        else:
            subset = data
        return subset[subset['label'].apply(predicate)].reset_index()

    @staticmethod
    def _load_wav(path):
        return torchaudio.load(path)[0]

    @staticmethod
    def _load_spec(path):
        return torch.load(path)

    def __getitem__(self, n: int) -> Tuple[Tensor, str]:
        label = self.data['label'][n]
        path = os.path.join(self.root, self.data["path"][n])
        return self._load_method(path), label

    def __len__(self) -> int:
        return len(self.data)

    @property
    def labels(self) -> List[str]:
        return self._labels


class MultiDataset(Dataset):
    """Combines several WalkerDatasets into one instance.
    Is used to train multilingual embeddings.
    Stacks all labels of given datasets into one list"""

    @property
    def unknown_index(self) -> Optional[int]:
        return None

    @property
    def labels(self) -> List[str]:
        return self._labels

    def __init__(self, datasets: List[Dataset]):
        super().__init__(datasets[0].is_wav)
        if not datasets:
            raise ValueError('Empty datasets creation is not allowed.')
        self._datasets = datasets
        self._lengths = [len(dataset) for dataset in datasets]
        self._len = sum(self._lengths)
        self._labels = sum([dataset.labels for dataset in datasets], [])

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, n: int) -> Tuple[Tensor, str]:
        if n >= self._len or n < 0:
            raise IndexError(f'Could not produce item of index {n}')
        index: int = 0
        aggregated: int = 0
        while index < len(self._lengths) and n - aggregated >= self._lengths[index]:
            aggregated += self._lengths[index]
            index += 1
        return self._datasets[index][n - aggregated]


class TransformedDataset(Dataset):
    """
    Dataset wrapper which uses given Transformer instance
    to create spectrogram and augment samples of given dataset
    """

    def __init__(self, dataset: Dataset, transformer: Transformer):
        super().__init__(is_wav=False)
        self.dataset = dataset
        self.transformer = transformer

    def __getitem__(self, n: int) -> Tuple[Tensor, str]:
        spectrogram: Tensor
        label: str
        if self.dataset.is_wav:
            waveform, label = self.dataset[n]
            spectrogram = self.transformer.to_mel_spectrogram(waveform)
        else:
            spectrogram, label = self.dataset[n]
        augmented: Tensor = self.transformer.augment(spectrogram)
        return augmented, label

    def __len__(self) -> int:
        return len(self.dataset)

    @property
    def unknown_index(self) -> Optional[int]:
        return self.dataset.unknown_index

    @property
    def labels(self) -> List[str]:
        return self.dataset.labels


class TargetProbaFsDataset(Dataset):
    """
    This Dataset implementation produces batches which are filled with few-shot target examples
    with given probability.
    Logic of picking target from the audio pool is all on (non_)target_dataset implementations,
    this class is just a wrapper around them.
    """

    def __init__(self, target_dataset: Dataset, non_target_dataset: Dataset, target: str,
                 target_proba: float):
        super().__init__(target_dataset.is_wav)
        self.target_dataset = target_dataset
        self.non_target_dataset = non_target_dataset
        self.target_proba = target_proba
        self.target = target

    def __getitem__(self, n: int) -> Tuple[Tensor, str]:
        if happen(self.target_proba):
            data, label = self.target_dataset[n % len(self.target_dataset)]
        else:
            data, label = self.non_target_dataset[n]
        if label == self.target:
            label = 'target'
        else:
            label = '_unknown'
        return data, label

    def __len__(self) -> int:
        return len(self.non_target_dataset)

    @property
    def unknown_index(self) -> Optional[int]:
        return 0

    @property
    def labels(self) -> List[str]:
        return ["_unknown", "target"]


class SampledDataset(Dataset):

    def __init__(self, dataset: Dataset, indices: List[int]):
        super().__init__(is_wav=dataset.is_wav)
        self._dataset = dataset
        self._indices = indices

    def __getitem__(self, n: int) -> Tuple[Tensor, str]:
        return self._dataset[self._indices[n]]

    def __len__(self) -> int:
        return len(self._indices)

    @property
    def unknown_index(self) -> Optional[int]:
        return self._dataset.unknown_index

    @property
    def labels(self) -> List[str]:
        return self._dataset.labels

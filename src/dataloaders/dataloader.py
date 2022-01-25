from abc import ABC, abstractmethod
from enum import Enum
from typing import Tuple

import torch


class DataLoaderMode(Enum):
    """Specifies the part of dataset to use in loader."""
    TRAINING = 0
    VALIDATION = 1
    TESTING = 2


class DataLoader(ABC):
    """
    Base class for every data loading entity in the project.
    Must provide load_data and get_batch functions to be used while learning.
    I haven't came up with beautiful structure of this one yet,
    so it's an interface with get_batch function for now.
    """

    @abstractmethod
    def get_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Produces a portion (batch) of data to feed into the Model instance.
        :return: tuple of (X, y), where X is the input data batch and y is the labels batch
        """

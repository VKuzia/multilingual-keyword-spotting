from abc import ABC, abstractmethod

import torch
from typing import Tuple


class DataLoader(ABC):
    """
    Base class for every data loading entity in the project.
    Must provide load_data and get_batch functions to be used while learning.
    I haven't came up with beautiful structure of this one yet, so it's an interface with get_batch function for now.
    """

    @abstractmethod
    def get_batch(self, batch_size: int, cuda: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Produces a portion (batch) of data to feed into the Model instance.

        :param batch_size: first dimension of batch tensor (number of input samples)
        :param cuda: shows transfer data to cuda or not
        :return: tuple of (X, y), where X is the input data batch and y is the corresponding labels batch
        """
        pass

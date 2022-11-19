import os
import random
from typing import Any, Optional, List, Dict

import numpy as np


def inspect_keys(dictionary: Optional[Dict[Any, Any]], keys: List[Any]):
    """Ensures all specified keys are present in dictionary. Otherwise throws KeyError"""
    if dictionary is None:
        if keys:
            raise ValueError(f'Trying to use empty dict with non-empty keys')
        return
    for item in keys:
        if item not in dictionary.keys():
            raise KeyError(f'There is no {item} in dictionary keys')


def happen(probability: float) -> bool:
    """is True with given probability"""
    return random.random() < probability


def rand_indices(high: int, size: int) -> List[int]:
    return list(np.random.random_integers(low=0, high=high, size=(size,)))


def dir_path(path: str) -> str:
    """argparse crutch to check whether given path is a directory"""
    if os.path.isdir(path):
        return path
    else:
        raise ValueError(f'Specified "{path}" is not a valid path to a dir')

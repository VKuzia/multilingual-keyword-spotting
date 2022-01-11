from __future__ import annotations

import json
from abc import abstractmethod
from typing import Any, Dict, Optional


class Config:
    """
    A set of json-like fields to be used in a dictionary-like manner.
    """

    @property
    @abstractmethod
    def default_dict(self) -> Dict[str, Optional[Any]]:
        """
        Specifies all possible keys for config, provides their defaults.
        Every child is supposed to be a set of independent rules, so there is no
        need in inheriting such dicts.
        """
        pass

    def __init__(self):
        self.data = self.default_dict.copy()

    def load_json(self, path: str) -> Config:
        """
        Loads configuration file using specified path, tries to map it into a dict.
        :param path: path to json file to parse
        :return: configuration dictionary-like object
        """
        with open(path) as json_file:
            json_data: Dict[str, Any] = json.load(json_file)
        for key, value in json_data.items():
            if key not in self.data.keys():
                raise ValueError(f'Unknown config parameter "{key}"')
            self.data[key] = value
        return self

    def __getitem__(self, item) -> Any:
        return self.data[item]


class TrainingConfig(Config):
    default_dict: Dict[str, Optional[Any]] = {
        'model_name': None,
        'model_version': None,
        'load_model_from_file': None,
        'load_optimizer_from_file': None,
        'checkpoint_version': None,
        'optimizer': None,
        'optimizer_parameters': None,
        'batch_size': None,
        'batches_per_epoch': None,
        'batches_per_validation': None,
        'epochs': None,
        'specific': None,
    }

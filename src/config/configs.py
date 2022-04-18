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
    def _default_dict(self) -> Dict[str, Optional[Any]]:
        """
        Specifies all possible keys for config, provides their defaults.
        Every child is supposed to be a set of independent rules, so there is no
        need in inheriting such dicts.
        """
        pass

    def __init__(self, additional_keys: Dict[str, Any] = None):
        self.data = self._default_dict.copy()
        if additional_keys:
            self.data.update(additional_keys)

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

    def __iter__(self):
        return iter(self.data.items())

    def __getitem__(self, item) -> Any:
        return self.data[item]

    def __len__(self) -> int:
        return len(self.data)


class TrainingConfig(Config):
    _default_dict: Dict[str, Optional[Any]] = {
        'model_name': None,
        'model_class': None,
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
        'save_after_epochs_count': None
    }

    def __init__(self, additional_keys: Dict[str, Any] = None):
        super().__init__(additional_keys)


class ValidationConfig(Config):
    _default_dict: Dict[str, Optional[Any]] = {
        'model_name': None,
        'model_class': None,
        'model_version': None,
        'load_model_from_file': None,
        'checkpoint_version': None,
        'batch_size': None,
        'batches_per_validation': None,
        'specific': None,
    }

    def __init__(self, additional_keys: Dict[str, Any] = None):
        super().__init__(additional_keys)

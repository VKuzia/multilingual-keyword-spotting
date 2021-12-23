from __future__ import annotations

import json
from abc import abstractmethod
from typing import Any, Dict, Optional


class Config:

    @property
    @abstractmethod
    def default_dict(self):
        pass

    def __init__(self):
        self.data = self.default_dict.copy()

    def load_json(self, path: str) -> Config:
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

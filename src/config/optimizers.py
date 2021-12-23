from typing import Optional, Dict, Any

import torch.optim

from src.models.model import Model
from src.utils.decorators import no_none
from src.utils.dicts import inspect_keys


@no_none
def build_optimizer(model: Model, name: str,
                    params: Optional[Dict[str, Any]]) -> torch.optim.Optimizer:
    if name == 'SGD':
        inspect_keys(params, ['learning_rate'])
        return torch.optim.SGD(model.kernel.parameters(), lr=params['learning_rate'])
    else:
        raise ValueError(f'Unknown optimizer type {name}.')

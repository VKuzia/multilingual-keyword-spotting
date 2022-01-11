from typing import Optional, Dict, Any

import torch.optim

from src.models import Model
from src.utils import no_none
from src.utils import inspect_keys


@no_none
def build_optimizer(model: Model, name: str,
                    params: Optional[Dict[str, Any]]) -> torch.optim.Optimizer:
    """
    Creates an instance of torch.optim.Optimizer using parameters provided.
    Is used in parsing configs.
    """
    if name == 'SGD':
        inspect_keys(params, ['learning_rate'])
        return torch.optim.SGD(model.kernel.parameters(), lr=params['learning_rate'])
    else:
        raise ValueError(f'Unknown optimizer type {name}.')

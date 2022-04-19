from typing import Optional, Dict, Any, Type

import torch.optim

from src.models import Model
from src.utils import no_none
from src.utils import inspect_keys


@no_none()
def build_optimizer(model: Model, name: str,
                    params: Optional[Dict[str, Any]],
                    output_only: bool = False) -> torch.optim.Optimizer:
    """
    Creates an instance of torch.optim.Optimizer using parameters provided.
    Is used in parsing configs.
    """
    model_params = model.kernel.parameters() if not output_only else model.kernel.output.parameters()
    if name == 'SGD':
        inspect_keys(params, ['learning_rate'])
        return torch.optim.SGD(model_params, lr=params['learning_rate'])
    elif name == 'Adam':
        inspect_keys(params, ['learning_rate'])
        return torch.optim.Adam(model_params, lr=params['learning_rate'])
    else:
        raise ValueError(f'Unknown optimizer type {name}.')


@no_none()
def get_optimizer_class(name: str) -> Type[torch.optim.Optimizer]:
    """    Returns optimizers class by given str name    """
    if name == 'SGD':
        return torch.optim.SGD
    elif name == 'Adam':
        return torch.optim.Adam
    else:
        raise ValueError(f'Unknown optimizer type {name}.')

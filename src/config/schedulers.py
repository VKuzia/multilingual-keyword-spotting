import torch.optim.lr_scheduler

from src.utils import no_none


@no_none()
def get_scheduler_class(name: str):
    if name == 'ExponentialLR':
        return torch.optim.lr_scheduler.ExponentialLR
    elif name == 'ReduceLROnPlateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau
    else:
        raise ValueError(f'Unknown scheduler type {name}.')

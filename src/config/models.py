from typing import Type

from src.models import Model, CoreModel, FewShotModel, CnnModel, CnnXModel, CnnYModel
from src.utils import no_none


@no_none()
def get_model_class(model_str: str) -> Type[Model]:
    if model_str == 'Core':
        return CoreModel
    elif model_str == 'Fs':
        return FewShotModel
    elif model_str == 'Cnn':
        return CnnModel
    elif model_str == 'CnnX':
        return CnnXModel
    elif model_str == 'CnnY':
        return CnnYModel
    else:
        raise ValueError(f'Unknown model class {model_str}.')

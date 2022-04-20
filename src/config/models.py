from typing import Type

from src.models import Model, CoreModel, FewShotModel
from src.utils import no_none


@no_none()
def get_model_class(model_str: str) -> Type[Model]:
    if model_str == 'Core':
        return CoreModel
    elif model_str == 'CoreFs':
        return FewShotModel
    else:
        raise ValueError(f'Unknown model class {model_str}.')

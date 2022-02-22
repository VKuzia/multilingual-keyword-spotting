from dataclasses import dataclass
from collections.abc import Iterable


def no_none(iterable_ok: bool = False):
    """Doesn't allow function to have None args. If None is given, throws ValueError"""

    def decorator(func):
        def function(*args, **kwargs):
            if any(not (iterable_ok and isinstance(arg, Iterable)) and arg is None for arg in args):
                raise ValueError(f'function {func.__name__}: does not accept None args.')
            return func(*args, **kwargs)

        return function

    return decorator


def no_none_dataclass(iterable_ok: bool = False):
    """Returns dataclass which doesn't allow its field to be None (throws ValueError)"""

    def dataclass_decorator(original_class):
        result = dataclass(original_class)
        original_init = result.__init__

        @no_none(iterable_ok)
        def __init__(self, *args, **kws):
            original_init(self, *args, **kws)

        result.__init__ = __init__
        return result

    return dataclass_decorator

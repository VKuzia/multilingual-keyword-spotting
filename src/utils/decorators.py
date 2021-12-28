from dataclasses import dataclass


def no_none(f):
    """Doesn't allow function to have None args. If None is given, throws ValueError"""

    def function(*args, **kwargs):
        if any(arg is None for arg in args):
            raise ValueError(f'function {f.__name__}: does not accept None args.')
        return f(*args, **kwargs)

    return function


def no_none_dataclass(original_class):
    """Returns dataclass which doesn't allow its field to be None (throws ValueError)"""

    result = dataclass(original_class)
    original_init = result.__init__

    @no_none
    def __init__(self, *args, **kws):
        original_init(self, *args, **kws)

    result.__init__ = __init__
    return result

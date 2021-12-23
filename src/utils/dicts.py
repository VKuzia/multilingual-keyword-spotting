from typing import Any, Iterable, Optional, List, Dict


def inspect_keys(dictionary: Optional[Dict[Any, Any]], keys: List[Any]):
    if dictionary is None:
        if keys:
            raise ValueError(f'Trying to use empty dict with non-empty keys')
        return
    for item in keys:
        if item not in dictionary.keys():
            raise KeyError(f'There is no {item} in dictionary keys')

from typing import Any, Optional, List, Dict


def inspect_keys(dictionary: Optional[Dict[Any, Any]], keys: List[Any]):
    """Ensures all specified keys are present in dictionary. Otherwise throws KeyError"""
    if dictionary is None:
        if keys:
            raise ValueError(f'Trying to use empty dict with non-empty keys')
        return
    for item in keys:
        if item not in dictionary.keys():
            raise KeyError(f'There is no {item} in dictionary keys')

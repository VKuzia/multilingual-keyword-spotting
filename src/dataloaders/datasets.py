from typing import Callable, Optional

from src.dataloaders import WalkerDataset, DataLoaderMode


def is_word_predicate(word: str) -> Callable[[str], bool]:
    """Predicate to check whether given short path to audio targets word provided"""
    return lambda x: x.split('/')[0] == word


class MonoMSWCDataset(WalkerDataset):
    """WalkerDataset implementation for MSWC dataset. Works with single language."""

    def __init__(self, path: str, language: str, subset: DataLoaderMode, part: str,
                 is_wav: bool = True,
                 predicate: Callable[[str], bool] = lambda label: True):
        self.language = language
        self.part = part
        super().__init__(f"{path}{language}/", subset, is_wav, predicate)

    @property
    def path_to_splits(self) -> str:
        return f"{self.language}_{self.part}.csv"

    @property
    def path_to_clips(self) -> str:
        return "clips/" if self.is_wav else "clips_tensors_1/"

    @property
    def unknown_index(self) -> Optional[int]:
        return None

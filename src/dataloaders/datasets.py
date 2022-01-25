import os
from typing import Callable, List, Optional

from src.dataloaders.base import WalkerDataset, DataLoaderMode


def is_word_predicate(word: str) -> Callable[[str], bool]:
    return lambda x: x.split('/')[0] == word


class MonoMSWCDataset(WalkerDataset):

    def __init__(self, path: str, language: str, subset: DataLoaderMode,
                 predicate: Callable[[str], bool] = lambda label: True):
        self.language = language
        super().__init__(f"{path}{language}/", subset, predicate)
        self._labels = os.listdir(f"{self.root}clips/")

    @property
    def train_list(self) -> str:
        return f"{self.language}_train.txt"

    @property
    def validation_list(self) -> str:
        # TODO: VALIDATION
        return f"{self.language}_train.txt"

    @property
    def test_list(self) -> str:
        return f"{self.language}_test.txt"

    @property
    def path_to_clips(self) -> str:
        return "clips/"

    def extract_label_short(self, path_to_clip: str) -> str:
        return path_to_clip.split('/')[0]

    def extract_label_full(self, path_to_clip: str) -> str:
        return path_to_clip.split('/')[-2]

    @property
    def labels(self) -> List[str]:
        return self._labels

    @property
    def unknown_index(self) -> Optional[int]:
        return None

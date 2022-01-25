import os
from typing import Tuple, Callable, List

import torch
import torchaudio

from torch import Tensor

from src.dataloaders import DataLoader, DataLoaderMode


def is_word_predicate(word: str) -> Callable[[str], bool]:
    return lambda x: x.split('/')[0] == word


class MonoMSWCDataset:

    def __init__(self, path: str, language: str, subset: DataLoaderMode,
                 predicate: Callable[[str], bool] = lambda label: True):
        self.path = f"{path}{language}/"
        self.language = language
        self.subset = subset
        self.labels = os.listdir(f"{self.path}clips/") + ["unknown"]

        if subset == DataLoaderMode.TRAINING:
            self.walker = self.load_list(f"{language}_train.txt", predicate)
        elif subset == DataLoaderMode.TESTING:
            self.walker = self.load_list(f"{language}_test.txt", predicate)
        else:
            raise ValueError(f"Can't handle unknown DataLoaderMode '{subset.name}'")

    def load_list(self, filename: str,
                  predicate: Callable[[str], bool] = lambda label: True) -> List[str]:
        filepath = os.path.join(self.path, filename)
        with open(filepath) as file:
            result = []
            for line in file:
                if predicate(self.line_to_label(line)):
                    result.append(os.path.join(self.path + "clips/", line.strip()))
            return result

    @staticmethod
    def line_to_label(line: str) -> str:
        return line.split('/')[0]

    def __getitem__(self, n: int) -> Tuple[Tensor, int, str]:
        label = self.walker[n].split('/')[-2]
        waveform, sample_rate = torchaudio.load(self.walker[n])
        return waveform, sample_rate, label

    def __len__(self) -> int:
        return len(self.walker)


class MonoMSWCDataLoader(DataLoader):
    transform = torchaudio.transforms. \
        Spectrogram(n_fft=97, hop_length=400)  # creates spectrograms of 1x49x40

    def __init__(self, path: str, language: str, mode: DataLoaderMode, batch_size: int,
                 cuda: bool = True):
        self.dataset = MonoMSWCDataset(path, language, mode)
        self.labels = self.dataset.labels
        self.batch_size = batch_size
        self.loader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
            pin_memory=cuda
        )
        self.cuda = cuda

    def get_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return next(self._yield_batch())

    def _yield_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Takes next batch of the self.loader, transforms it and yields."""
        for data, labels in self.loader:
            transformed_data = self.transform(data).repeat(1, 3, 1, 1)
            if self.cuda:
                yield transformed_data.to('cuda'), labels.to('cuda')
            else:
                yield transformed_data, labels

    def label_to_index(self, word: str) -> int:
        """Returns the index of the given word"""
        if word in self.labels:
            return self.labels.index(word)
        else:
            return len(self.labels) - 1  # unknown label

    def collate_fn(self, batch):
        """Creates a batch of samples"""
        tensors, targets = [], []

        # initial data tuple has the form: waveform, sample_rate, label, speaker_id, utter_number
        for waveform, _, label in batch:
            tensors += [waveform]
            targets += [torch.tensor(self.label_to_index(label))]

        # Group the list of tensors into a batched tensor
        tensors = self.pad_sequence(tensors)
        targets = torch.stack(targets)

        return tensors, targets

    @staticmethod
    def pad_sequence(batch):
        """Make all tensor in a batch the same length by padding with zeros"""
        batch = [item.t() for item in batch]
        batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.)
        return batch.permute(0, 2, 1)

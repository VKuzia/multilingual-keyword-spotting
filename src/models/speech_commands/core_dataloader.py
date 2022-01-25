from typing import Tuple

import torch


class CoreDataLoader(SpeechCommandsBase):
    """
    Implements DataLoader interface for SpeechCommands dataset.
    Mainly taken from
    https://pytorch.org/tutorials/intermediate/speech_command_recognition_with_torchaudio.html
    """

    def __init__(self, path: str, mode: DataLoaderMode, batch_size: int, cuda: bool = True):
        super().__init__()
        self.dataset = SpeechCommandsDataset(path, mode)
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

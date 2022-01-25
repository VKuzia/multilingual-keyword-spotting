from typing import Tuple

import torch
from src.models.speech_commands.core_dataloader import SpeechCommandsBase, SpeechCommandsDataset, \
    DataLoaderMode


class FewShotSpeechCommandsDataLoader(SpeechCommandsBase):
    """
    Implementation of SpeechCommands dataset wrapper featured to work with few-shot models.
    Accepts self.target as a word to be spotted and shuffles data in a way self.target_probability
    of every batch is targets.
    Based on SpeechCommandsDataset and
    https://pytorch.org/tutorials/intermediate/speech_command_recognition_with_torchaudio.html
    """

    def __init__(self, target: str, path: str, mode: DataLoaderMode, batch_size: int,
                 target_probability: float,
                 cuda: bool = True):
        super().__init__()
        self.target = target
        self.dataset_targets = SpeechCommandsDataset(path, mode, lambda label: target == label)
        self.dataset_non_targets = SpeechCommandsDataset(path, mode, lambda label: target != label)
        self.batch_size = batch_size
        self.cuda = cuda
        target_batch_size = int(batch_size * target_probability)
        non_target_batch_size = batch_size - target_batch_size
        self.target_loader = torch.utils.data.DataLoader(
            self.dataset_targets,
            batch_size=target_batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
            pin_memory=cuda
        )
        self.non_target_loader = torch.utils.data.DataLoader(
            self.dataset_non_targets,
            batch_size=non_target_batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
            pin_memory=cuda
        )

    def label_to_index(self, word: str) -> int:
        return 1 if word == self.target else 0

    def get_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return next(self._yield_batch())

    def _yield_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        for (target_data, target_labels), (non_target_data, non_target_labels) in zip(
                self.target_loader,
                self.non_target_loader):
            data = torch.concat((target_data, non_target_data), dim=0)
            labels = torch.concat((target_labels, non_target_labels), dim=0)
            transformed_data = self.transform(data).repeat(1, 3, 1, 1)
            if self.cuda:
                yield transformed_data.to('cuda'), labels.to('cuda')
            else:
                yield transformed_data, labels

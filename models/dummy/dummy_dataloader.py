import torch

from dataloaders.dataloader import DataLoader


class DummyDataLoader(DataLoader):
    """
    Example of a DataLoader implementation. Produces random noise.
    """

    def get_batch(self, batch_size: int, cuda: bool = True):
        return torch.rand((batch_size, 128)).to('cuda' if cuda else 'cpu'), \
               torch.rand(batch_size).to(dtype=torch.long).to('cuda' if cuda else 'cpu')

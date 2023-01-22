import numpy as np
import torch
from tqdm import trange

from src import models

import src.dataloaders as data
from sklearn.metrics import confusion_matrix


def run_through(module: models.Module, loader: data.DataLoader, enable_tqdm: bool = True, cuda: bool = False) -> (
        torch.Tensor, torch.Tensor, torch.Tensor):
    outputs = []
    preds = []
    labels = []
    device = torch.device('cpu' if not cuda else 'cuda:0')
    with torch.no_grad():
        range_ = trange(loader.get_batch_count()) if enable_tqdm else range(
            loader.get_batch_count())
        for _ in range_:
            data_batch, labels_batch = loader.get_batch()
            model_output = module(data_batch).to(device)
            outputs.append(model_output)
            preds.append(model_output.argmax(dim=1))
            labels.append(labels_batch.to(device))
    return torch.concat(outputs, 0), torch.concat(preds, 0), torch.concat(labels, 0)


def build_confusion_matrix(preds: torch.Tensor, labels: torch.Tensor) -> np.array:
    return confusion_matrix(labels, preds)

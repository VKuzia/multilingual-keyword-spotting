from typing import List

import torch

from src import models

import src.dataloaders as data


class Validator:

    def __init__(self):
        self.outputs = None
        self.labels = None
        self.preds = None

    def accept(self, model: models.Module, loader: data.DataLoader):
        outputs = []
        labels = []
        preds = []
        with torch.no_grad():
            for _ in range(loader.get_batch_count()):
                data_batch, labels_batch = loader.get_batch()
                model_output = model(data_batch)
                outputs.append(model_output)
                preds.append(model_output.argmax(dim=1))
                labels.append(labels_batch)
        self.outputs = torch.stack(outputs)
        self.labels = torch.stack(labels)
        self.preds = torch.stack(labels)

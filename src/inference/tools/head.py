from typing import Dict, Any

import torch.nn.functional

import src.models as models


class BinaryLabeledHead(models.Module):

    def __init__(self, config, model_io: models.ModelIO, root: str):
        super().__init__()
        self.name = config['name']
        self.threshold = config['threshold']
        self.params = config['params']
        self.model = model_io.build_module(self.params, root)

    def get_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'threshold': self.threshold,
            'params': self.params
        }

    def infer(self, x) -> bool:
        output = self.model(x)
        output = torch.nn.functional.softmax(output[0], dim=0)[1].item()
        return output > self.threshold

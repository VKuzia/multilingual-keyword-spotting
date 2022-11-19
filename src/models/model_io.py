import copy
import json
import os
import shutil

import torch
import src.models.new_models as models
import src.models.model as md


class ModelIO:
    EMBEDDING_STATE_DICT_NAME = "embedding.pth"
    HEAD_STATE_DICT_NAME = "head.pth"
    OPTIMIZER_STATE_DICT_NAME = "optimizer.pth"
    SCHEDULER_STATE_DICT_NAME = "scheduler.pth"
    LEARNING_INFO_NAME = "learning_info.json"
    CONFIG_NAME = "config.json"

    def __init__(self, root: str):
        self.root = root

    def build_model(self, config) -> md.Model:
        """
        Provides the ready-to-go Model instance according to config.
        All checkpoint loading and pytorch elements building structure is encapsulated here.
        Config must provide full descriptions of 'model' (both 'embedding' and 'head'),
        'optimizer', 'scheduler' and 'loss' instances to use.
        """
        kernel = models.TotalKernel(self._build_module(config['model']['embedding'], self.root),
                                    self._build_module(config['model']['head'], self.root))
        optimizer = self._build_optimizer(config['optimizer'], kernel, self.root)
        scheduler = self._build_scheduler(config['scheduler'], optimizer, self.root)
        loss = self._build_loss(config['loss'])
        return md.Model(kernel, optimizer, scheduler, loss)

    def save_model(self, config, model: md.Model, output_dir: str, full_path: bool = True):
        dir_path = os.path.join(self.root, output_dir) if not full_path else output_dir
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
        os.makedirs(dir_path)
        torch.save(model.kernel.embedding.state_dict(),
                   os.path.join(dir_path, self.EMBEDDING_STATE_DICT_NAME))
        torch.save(model.kernel.head.state_dict(),
                   os.path.join(dir_path, self.HEAD_STATE_DICT_NAME))
        torch.save(model.optimizer.state_dict(),
                   os.path.join(dir_path, self.OPTIMIZER_STATE_DICT_NAME))
        torch.save(model.scheduler.state_dict(),
                   os.path.join(dir_path, self.SCHEDULER_STATE_DICT_NAME))
        with open(os.path.join(dir_path, self.LEARNING_INFO_NAME), 'w') as learning_file:
            json.dump(model.learning_info.to_dict(), learning_file, indent=2)
        with open(os.path.join(dir_path, self.CONFIG_NAME), 'w') as config_file:
            json.dump(config, config_file, indent=2)

    @staticmethod
    def _build_module(config, root: str) -> models.Module:
        if config['name'] == 'efficient_net':
            module = models.EfficientNetKernel(config['output'], config['hidden'])
        elif config['name'] == 'softmax':
            module = models.SoftmaxHeadKernel(config['input'], config['output'])
        else:
            raise ValueError(f'Model "{config["name"]}" is unknown.')
        if config.get('path'):
            module.load_state_dict(torch.load(os.path.join(root, config['path'])))
        if config.get('freeze', False):
            for param in module.parameters():
                param.requires_grad = False
        return module

    @staticmethod
    def _build_optimizer(config, model: models.Module, root: str) -> torch.optim.Optimizer:
        config_copy = copy.deepcopy(config)
        del config_copy['name']
        if config_copy.get('path'):
            del config_copy['path']
        if config['name'] == 'SGD':
            optimizer = torch.optim.SGD(model.parameters(), **config_copy)
        elif config['name'] == 'Adam':
            optimizer = torch.optim.Adam(model.parameters(), **config_copy)
        else:
            raise ValueError(f'Optimizer "{config["name"]} is unknown."')
        if config.get('path'):
            optimizer.load_state_dict(torch.load(os.path.join(root, config['path'])))
        return optimizer

    @staticmethod
    def _build_scheduler(config, optimizer: torch.optim.Optimizer, root: str):
        config_copy = copy.deepcopy(config)
        del config_copy['name']
        if config_copy.get('path'):
            del config_copy['path']
        if config['name'] == 'exponential':
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, **config_copy)
        else:
            raise ValueError(f'Scheduler "{config["name"]}" is unknown.')
        if config.get('path'):
            scheduler.load_state_dict(torch.load(os.path.join(root, config['path'])))
        return scheduler

    @staticmethod
    def _build_loss(config):
        if config['name'] == 'xent':
            return torch.nn.NLLLoss()
        else:
            raise ValueError(f'Loss "{config["name"]}" is unknown.')

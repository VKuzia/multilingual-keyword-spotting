import json
import os
import shutil
from typing import Type, Any, Dict, Optional

import torch
from torch import nn

from src.models import Model, ModelInfoTag, ModelCheckpoint, ModelLearningInfo


class ModelIO:
    KERNEL_STATE_DICT_NAME = "kernel_state.pth"
    OPTIMIZER_STATE_DICT_NAME = "optimizer_state.pth"
    SCHEDULER_STATE_DICT_NAME = "scheduler.pth"
    INFO_NAME = "model_info.json"
    LEARNING_INFO_NAME = "learning_info.json"

    def __init__(self, base_path: str = "saved_models/"):
        self.base_path = base_path

    def load_kernel(self, module_class, model_info: ModelInfoTag, checkpoint_version: int, *,
                    kernel_args: Optional[Dict[str, Any]] = None,
                    load_output_layer: bool = True) -> nn.Module:
        checkpoint_path = self._get_dir(model_info, checkpoint_version)
        kernel: nn.Module = module_class(**kernel_args)
        if load_output_layer:
            state_dict = torch.load(f"{checkpoint_path}/{self.KERNEL_STATE_DICT_NAME}")
            kernel.load_state_dict(state_dict)
        else:
            print(checkpoint_path)
            print(self.base_path)
            print(model_info.name)
            kernel.load_state_dict(
                self._cut_output_layer(f"{checkpoint_path}/{self.KERNEL_STATE_DICT_NAME}"))
        return kernel

    def load_optimizer(self, optimizer_class, module, model_info: ModelInfoTag,
                       checkpoint_version: int, *,
                       optimizer_args: Optional[Dict[str, Any]]) -> torch.optim.Optimizer:
        checkpoint_path = self._get_dir(model_info, checkpoint_version)
        optimizer = optimizer_class(module.parameters(), **optimizer_args)
        optimizer.load_state_dict(
            torch.load(f'{checkpoint_path}/{self.OPTIMIZER_STATE_DICT_NAME}'))
        return optimizer

    def load_scheduler(self, scheduler_class, optimizer, model_info: ModelInfoTag,
                       checkpoint_version: int, *,
                       scheduler_args: Optional[Dict[str, Any]]) -> torch.optim.Optimizer:
        checkpoint_path = self._get_dir(model_info, checkpoint_version)
        scheduler = scheduler_class(optimizer, **scheduler_args)
        scheduler.load_state_dict(
            torch.load(f'{checkpoint_path}/{self.SCHEDULER_STATE_DICT_NAME}'))
        return scheduler

    def load_model(self, model_class: Type[Model], model_info: ModelInfoTag,
                   checkpoint_version: int,
                   optimizer_class, scheduler_class, *,
                   kernel_args: Optional[Dict[str, Any]],
                   optimizer_args: Optional[Dict[str, Any]],
                   scheduler_args: Optional[Dict[str, Any]],
                   load_output_layer: bool = True,
                   output_only: bool = False,
                   cuda: bool = True) -> Model:
        directory = self._get_dir(model_info, checkpoint_version)
        kernel = self.load_kernel(model_class.get_kernel_class(), model_info, checkpoint_version,
                                  kernel_args=kernel_args, load_output_layer=load_output_layer)
        optimizer = None
        if optimizer_class:
            if not output_only:
                optimizer = self.load_optimizer(optimizer_class, kernel, model_info,
                                                checkpoint_version, optimizer_args=optimizer_args)
            else:
                if cuda:
                    kernel.output = kernel.output.cuda()
                optimizer = self.load_optimizer(optimizer_class, kernel.output, model_info,
                                                checkpoint_version, optimizer_args=optimizer_args)
        scheduler = None
        if scheduler_class:
            if optimizer_class is None:
                raise ValueError("Can't load scheduler for None optimizer")
            scheduler = self.load_scheduler(scheduler_class, optimizer, model_info,
                                            checkpoint_version, scheduler_args=scheduler_args)

        with open(f"{directory}/{self.INFO_NAME}", "r") as info_file, open(
                f"{directory}/{self.LEARNING_INFO_NAME}", "r") as learning_file:
            info_dict: Dict[str, Any] = json.loads(info_file.read())
            info_tag: ModelInfoTag = ModelInfoTag(**info_dict)
            learning_dict: Dict[str, Any] = json.loads(learning_file.read())
            learning_info: ModelLearningInfo = ModelLearningInfo(**learning_dict)

        model: Model = model_class(kernel, optimizer, scheduler, model_class.get_loss_function(),
                                   info_tag, cuda)
        model.learning_info = learning_info
        return model

    def save_model(self, model):
        checkpoint: ModelCheckpoint = model.build_checkpoint()
        directory: str = self._get_dir(model.info_tag, model.checkpoint_id)
        try:
            if os.path.exists(directory):
                shutil.rmtree(directory)
            os.makedirs(directory)
        except OSError as error:
            print(error)
        torch.save(checkpoint.kernel_state_dict, f'{directory}/{self.KERNEL_STATE_DICT_NAME}')
        torch.save(checkpoint.optimizer_state_dict, f'{directory}/{self.OPTIMIZER_STATE_DICT_NAME}')
        torch.save(checkpoint.scheduler_state_dict, f'{directory}/{self.SCHEDULER_STATE_DICT_NAME}')

        with open(f"{directory}/{self.INFO_NAME}", "w") as info_file:
            info_file.write(json.dumps(checkpoint.info_tag, default=lambda o: o.__dict__, indent=1))
        with open(f"{directory}/{self.LEARNING_INFO_NAME}", "w") as info_file:
            info_file.write(
                json.dumps(checkpoint.learning_info, default=lambda o: o.__dict__, indent=1))

    def _get_dir(self, model_info: ModelInfoTag, checkpoint_version: int) -> str:
        """Constructs directory name according given model_info and checkpoint version."""
        return f"{self.base_path}/{model_info.name}_{model_info.version_tag}_[{checkpoint_version}]"

    @staticmethod
    def _cut_output_layer(path: str):
        state_dict = torch.load(path)
        k_pop = [key for key in state_dict.keys() if key.startswith('output')]
        for key in k_pop:
            state_dict.pop(key, None)
        return state_dict

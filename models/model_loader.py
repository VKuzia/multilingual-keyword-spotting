from typing import Optional, Type, Any, Dict

import torch
import json

from models.model import Model, ModelCheckpoint, build_model_of, ModelInfoTag, ModelLearningInfo
import os
import shutil

KERNEL_STATE_DICT_NAME = "kernel_state.pth"
OPTIMIZER_STATE_DICT_NAME = "optimizer_state.pth"
INFO_NAME = "info.json"
LEARNING_INFO_NAME = "learning.json"


class ModelIOHelper:
    """
    This class is able of loading and saving models using their info tags and class type.
    Uses base directory specified in constructor and naming logic using specified constants and get_dir method.

    TODO: correct IO exceptions handling
    """

    def __init__(self, base_path: str = "saved_models/"):
        self.base_path = base_path

    def load_model(self, model_class: Type[Model], model_info: ModelInfoTag, checkpoint_version: int) -> Model:
        """
        Constructs Model instance using locally saved checkpoint and model_info.
        :param model_class: class to construct instance of
        :param model_info: model's name parameters to specify model to load
        :param checkpoint_version: identifier of saved checkpoint
        :return: constructed model
        """
        path = self.get_dir(model_info, checkpoint_version)
        return self.load_model_by_path(model_class, path, checkpoint_version)

    def load_model_by_path(self, model_class: Type[Model], path: str, checkpoint_version: int = 0,
                           use_base_path: bool = True) -> Model:
        """
        Constructs Model instance according specified path and class type.
        Basically builds the default model and assigns loaded dictionaries to its parameters.
        :param model_class: class to construct instance of
        :param path: path to load model from
        :param checkpoint_version: identifier of saved checkpoint
        :param use_base_path: specifies whether to use self.base_path prefix for loading the model.
        :return: constructed model.

        TODO: correct IO exceptions handling
        """
        if use_base_path:
            path = self.base_path + path
        with open(f"{path}/{INFO_NAME}", "r") as info_file, open(f"{path}/{LEARNING_INFO_NAME}", "r") as learning_file:
            info_dict: Dict[str, Any] = json.loads(info_file.readline())
            info_tag: ModelInfoTag = ModelInfoTag(**info_dict)
            learning_dict: Dict[str, Any] = json.loads(learning_file.readline())
            learning_info: ModelLearningInfo = ModelLearningInfo(**learning_dict)

            model: Model = build_model_of(model_class, info_tag)
            model.kernel.load_state_dict(torch.load(f"{path}/{KERNEL_STATE_DICT_NAME}"))
            model.optimizer.load_state_dict(torch.load(f"{path}/{OPTIMIZER_STATE_DICT_NAME}"))
            model.learning_info = learning_info
            model.checkpoint_id = checkpoint_version
            return model

    def save_model(self, model: Model, path: Optional[str] = None, use_base_path: bool = True) -> None:
        """
        Saves model's checkpoint into the directory specified by self.get_dir logic.
        :param model: model to save
        :param path: path to save model to
        :param use_base_path: specifies whether to use self.base_path prefix for loading the model.

        TODO: correct IO exceptions handling
        """
        if use_base_path:
            path = self.base_path
        if path is None:
            raise ValueError("Couldn't save model as path is null.")
        checkpoint: ModelCheckpoint = model.build_checkpoint()
        dir_name: str = self.get_dir(checkpoint.info_tag, checkpoint.id)
        final_path: str = path + dir_name
        try:
            if os.path.exists(final_path):
                shutil.rmtree(final_path)
            os.makedirs(final_path)
        except OSError as error:
            print(error)
        torch.save(checkpoint.kernel_state_dict, f"{final_path}/{KERNEL_STATE_DICT_NAME}")
        torch.save(checkpoint.optimizer_state_dict, f"{final_path}/{OPTIMIZER_STATE_DICT_NAME}")
        with open(f"{final_path}/{INFO_NAME}", "w") as info_file:
            info_file.write(json.dumps(checkpoint.info_tag, default=lambda o: o.__dict__))
        with open(f"{final_path}/{LEARNING_INFO_NAME}", "w") as info_file:
            info_file.write(json.dumps(checkpoint.learning_info, default=lambda o: o.__dict__))

    @staticmethod
    def get_dir(model_info: ModelInfoTag, checkpoint_version: int) -> str:
        """Constructs directory name according given model_info and checkpoint version."""
        return f"{model_info.name}_{model_info.version_tag}_[{checkpoint_version}]"

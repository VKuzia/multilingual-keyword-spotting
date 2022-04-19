from typing import List, Type

import torch
from torch import nn

from src.config import Config, build_optimizer
from src.config.models import get_model_class
from src.models import Model, ModelIOHelper, ModelInfoTag, build_model_of
from src.paths import PATH_TO_SAVED_MODELS


def build_default_fs_model(config: Config,
                           languages: List[str],
                           *,
                           validation: bool = False) -> (Model, ModelIOHelper):
    info_tag: ModelInfoTag = ModelInfoTag(config['model_name'], config['model_version'], languages,
                                          config['dataset_part'])
    model_io: ModelIOHelper = ModelIOHelper(PATH_TO_SAVED_MODELS)
    model: Model
    embedding_class = get_model_class(config['embedding_class'])
    model_class: Type[Model] = get_model_class(config['model_class'])
    if config['load_model_from_file']:
        if config['checkpoint_version'] is None:
            raise ValueError('Version of model to be loaded is unknown')
        model = model_io.load_model(model_class, info_tag, config['checkpoint_version'],
                                    kernel_args={"embedding_class": embedding_class,
                                                 "output_channels": None})
        if config['load_embedding']:
            raise ValueError("New embedding should be loaded only into new models")
    else:
        model: Model = build_model_of(model_class, info_tag,
                                      kernel_args={"embedding_class": embedding_class,
                                                   "output_channels": None})
        if config['load_embedding']:
            embedding_info_tag: ModelInfoTag = ModelInfoTag(config['embedding_name'],
                                                            config['embedding_version'],
                                                            [])
            if config['embedding_checkpoint_version'] is None:
                raise ValueError('Version of embedding to be loaded is unknown')
            embedding = model_io.load_model(embedding_class, embedding_info_tag,
                                            config['embedding_checkpoint_version'],
                                            kernel_args={"output_channels": None},
                                            load_output_layer=False)
            embedding.kernel.output = nn.Identity()
            for parameter in embedding.kernel.parameters():
                parameter.requires_grad = False
            model.kernel.core = embedding.kernel

    if not validation:
        if not config['load_optimizer_from_file']:
            optimizer: torch.optim.Optimizer = build_optimizer(model, config['optimizer'],
                                                               config['optimizer_parameters'],
                                                               output_only=True)
            model.optimizer = optimizer
    else:
        model.optimizer = None
    return model, model_io

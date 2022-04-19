from typing import List, Type

from src.config import Config
from src.config.models import get_model_class
from src.config.optimizers import get_optimizer_class
from src.config.schedulers import get_scheduler_class
from src.models import Model, ModelIOHelper, ModelInfoTag
from src.models.model_io import ModelIO
from src.paths import PATH_TO_SAVED_MODELS


def build_default_fs_model(config: Config,
                           languages: List[str],
                           cuda: bool = True) -> (Model, ModelIOHelper):
    info_tag: ModelInfoTag = ModelInfoTag(config['model_name'], config['model_version'], languages,
                                          config['dataset_part'])
    model_io: ModelIO = ModelIO(PATH_TO_SAVED_MODELS)
    embedding_class = get_model_class(config['embedding_class'])
    model_class: Type[Model] = get_model_class(config['model_class'])
    model: Model
    optimizer_class = get_optimizer_class(config['optimizer_class'])
    optimizer_params = config['optimizer_parameters']
    scheduler_class = get_scheduler_class(config['scheduler_class'])
    scheduler_params = config['scheduler_parameters']
    if config['load_model_from_file']:
        if config['load_embedding']:
            raise ValueError("New embedding should be loaded only into new models")
        if config['checkpoint_version'] is None:
            raise ValueError('Version of model to be loaded is unknown')
        if not config['load_optimizer_from_file']:
            optimizer_class = None
        if not config['load_scheduler_from_file']:
            scheduler_class = None
        model = model_io.load_model(model_class,
                                    info_tag,
                                    config['checkpoint_version'],
                                    optimizer_class,
                                    scheduler_class,
                                    kernel_args=dict(),
                                    optimizer_args=optimizer_params,
                                    scheduler_args=scheduler_params,
                                    output_only=True,
                                    cuda=cuda)
        if not config['load_optimizer_from_file']:
            optimizer_class = get_optimizer_class(config['optimizer_class'])
            model.optimizer = optimizer_class(model.kernel.output.parameters(),
                                              **config['optimizer_parameters'])
        if not config['load_scheduler_from_file']:
            scheduler_class = get_scheduler_class(config['scheduler_class'])
            model.scheduler = scheduler_class(model.optimizer, **config['scheduler_parameters'])

    else:
        if not config['load_embedding']:
            raise ValueError("There is no sense in a few-shot model over untrained embedding")
        if config['embedding_checkpoint_version'] is None:
            raise ValueError('Version of embedding to be loaded is unknown')
        embedding_info_tag: ModelInfoTag = ModelInfoTag(config['embedding_name'],
                                                        config['embedding_version'],
                                                        [],
                                                        [])
        embedding = model_io.load_model(embedding_class,
                                        embedding_info_tag,
                                        config['embedding_checkpoint_version'],
                                        None,
                                        None,
                                        kernel_args={"output_channels": None},
                                        optimizer_args=None,
                                        scheduler_args=None,
                                        load_output_layer=False,
                                        cuda=cuda)
        # embedding.kernel.output = nn.Identity()
        for parameter in embedding.kernel.parameters():
            parameter.requires_grad = False

        kernel_class = model_class.get_kernel_class()
        kernel_args = {"embedding": embedding.kernel}
        print(kernel_class)
        kernel = kernel_class(**kernel_args)
        optimizer = optimizer_class(kernel.output.parameters(), **optimizer_params)
        model: Model = model_class(kernel,
                                   optimizer,
                                   scheduler_class(optimizer, **scheduler_params),
                                   model_class.get_loss_function(),
                                   info_tag,
                                   cuda)
    return model, model_io


def build_default_fs_validation_model(config: Config,
                                      languages: List[str],
                                      cuda: bool = True) -> (Model, ModelIOHelper):
    info_tag: ModelInfoTag = ModelInfoTag(config['model_name'], config['model_version'], languages,
                                          config['dataset_part'])
    model_io: ModelIO = ModelIO(PATH_TO_SAVED_MODELS)
    model_class: Type[Model] = get_model_class(config['model_class'])
    model: Model
    if config['checkpoint_version'] is None:
        raise ValueError('Version of model to be loaded is unknown')
    model = model_io.load_model(model_class,
                                info_tag,
                                config['checkpoint_version'],
                                None,
                                None,
                                kernel_args=dict(),
                                optimizer_args=None,
                                scheduler_args=None,
                                output_only=True,
                                cuda=cuda)
    return model, model_io

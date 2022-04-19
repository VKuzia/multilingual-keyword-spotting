from typing import List, Optional, Callable, Type

from src.config import TrainingConfig, Config, get_model_class, get_optimizer_class, \
    get_scheduler_class
from src.dataloaders import DataLoader, DataLoaderMode, Dataset, MonoMSWCDataset, SpecDataset, \
    MultiDataset
from src.models import ModelInfoTag, Model, ModelIO
from src.paths import PATH_TO_SAVED_MODELS, PATH_TO_MSWC_WAV
from src.trainers import Trainer, TrainingParams, DefaultTrainer
from src.trainers.handlers import Printer, PrinterHandler, ClassificationValidator, ValidationMode, \
    ModelSaver
from src.transforms import DefaultTransformer


# This code is too dirty. No comments for now.
# TODO: cleanup and comment

def build_default_trainer(config: TrainingConfig,
                          train_loader: DataLoader,
                          validation_loader: DataLoader,
                          model_io: ModelIO) -> (Trainer, TrainingParams):
    printer: Printer = Printer(config['epochs'], config['batches_per_epoch'])
    printer_handler: PrinterHandler = PrinterHandler(printer)
    validation_validator: ClassificationValidator = \
        ClassificationValidator(validation_loader,
                                batch_count=config['batches_per_validation'],
                                mode=ValidationMode.VALIDATION)
    training_validator: ClassificationValidator = \
        ClassificationValidator(train_loader,
                                batch_count=config['batches_per_validation'],
                                mode=ValidationMode.TRAINING)
    trainer: Trainer = DefaultTrainer([], [printer_handler],
                                      [validation_validator, training_validator,
                                       ModelSaver(model_io, config['save_after_epochs_count']),
                                       printer_handler])

    training_params: TrainingParams = TrainingParams(batch_count=config['batches_per_epoch'],
                                                     batch_size=config['batch_size'],
                                                     epoch_count=config['epochs'])
    return trainer, training_params


def build_default_model(config: Config,
                        languages: List[str],
                        output_channels: Optional[int],
                        *,
                        cuda: bool = True) -> (Model, ModelIO):
    info_tag: ModelInfoTag = ModelInfoTag(config['model_name'], config['model_version'],
                                          languages, config['dataset_part'])
    model_io: ModelIO = ModelIO(PATH_TO_SAVED_MODELS)
    model_class: Type[Model] = get_model_class(config['model_class'])
    model: Model
    optimizer_class = get_optimizer_class(config['optimizer_class'])
    optimizer_params = config['optimizer_parameters']
    scheduler_class = get_scheduler_class(config['scheduler_class'])
    scheduler_params = config['scheduler_parameters']
    if config['load_model_from_file']:
        if config['checkpoint_version'] is None:
            raise ValueError('Version of model to be loaded is unknown')
        if not config['load_optimizer_from_file']:
            optimizer_class = None
        if not config['load_scheduler_from_file']:
            scheduler_class = None
        model: Model = model_io.load_model(model_class,
                                           info_tag,
                                           config['checkpoint_version'],
                                           optimizer_class,
                                           scheduler_class,
                                           kernel_args={"output_channels": output_channels},
                                           optimizer_args=optimizer_params,
                                           scheduler_args=scheduler_params)
        if not config['load_optimizer_from_file']:
            optimizer_class = get_optimizer_class(config['optimizer_class'])
            model.optimizer = optimizer_class(model.kernel.parameters(),
                                              **config['optimizer_parameters'])
        if not config['load_scheduler_from_file']:
            scheduler_class = get_scheduler_class(config['scheduler_class'])
            model.scheduler = scheduler_class(model.optimizer, **config['scheduler_parameters'])
    else:
        kernel_class = model_class.get_kernel_class()
        kernel_args = {"output_channels": output_channels}
        kernel = kernel_class(**kernel_args)
        optimizer = optimizer_class(kernel.parameters(), **optimizer_params)
        model: Model = model_class(kernel,
                                   optimizer,
                                   scheduler_class(optimizer, **scheduler_params),
                                   model_class.get_loss_function(),
                                   info_tag,
                                   cuda)
    return model, model_io


def build_default_validation_model(config: Config,
                                   languages: List[str],
                                   output_channels: Optional[int],
                                   *,
                                   cuda: bool = True) -> (Model, ModelIO):
    info_tag: ModelInfoTag = ModelInfoTag(config['model_name'], config['model_version'],
                                          languages, config['dataset_part'])
    model_io: ModelIO = ModelIO(PATH_TO_SAVED_MODELS)
    model_class: Type[Model] = get_model_class(config['model_class'])
    model: Model
    if config['checkpoint_version'] is None:
        raise ValueError('Version of model to be loaded is unknown')
    model: Model = model_io.load_model(model_class,
                                       info_tag,
                                       config['checkpoint_version'],
                                       None,
                                       None,
                                       kernel_args={"output_channels": output_channels},
                                       optimizer_args=None,
                                       scheduler_args=None)
    return model, model_io


def get_multi_dataset(config: Config, mode: DataLoaderMode,
                      predicate: Callable[[str], bool] = lambda x: True) -> Dataset:
    dataset_list: List[Dataset] = []
    for language in config['languages']:
        base = MonoMSWCDataset(PATH_TO_MSWC_WAV,
                               language,
                               mode,
                               is_wav=False,
                               part=config['dataset_part'],
                               predicate=predicate)
        dataset_list.append(SpecDataset(base, DefaultTransformer()))
    return MultiDataset(dataset_list)

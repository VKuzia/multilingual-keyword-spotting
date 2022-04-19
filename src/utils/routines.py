from typing import List, Optional, Callable

import torch

from src.config import TrainingConfig, build_optimizer, Config
from src.config.models import get_model_class
from src.dataloaders import DataLoader, DataLoaderMode, Dataset, MonoMSWCDataset, \
    ClassificationDataLoader
from src.dataloaders.base import SpecDataset, MultiDataset
from src.models import ModelIOHelper, ModelInfoTag, Model, build_model_of
from src.paths import PATH_TO_SAVED_MODELS, PATH_TO_MSWC_WAV
from src.trainers import Trainer, TrainingParams, DefaultTrainer
from src.trainers.handlers import Printer, PrinterHandler, ClassificationValidator, ValidationMode, \
    ModelSaver
from src.transforms.transformers import DefaultTransformer


def build_default_trainer(config: TrainingConfig,
                          train_loader: DataLoader,
                          validation_loader: DataLoader,
                          model_io: ModelIOHelper) -> (Trainer, TrainingParams):
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
                        validation: bool = False) -> (Model, ModelIOHelper):
    info_tag: ModelInfoTag = ModelInfoTag(config['model_name'], config['model_version'],
                                          languages, config['dataset_part'])
    model_io: ModelIOHelper = ModelIOHelper(PATH_TO_SAVED_MODELS)
    model_class = get_model_class(config['model_class'])
    model: Model
    if config['load_model_from_file']:
        if config['checkpoint_version'] is None:
            raise ValueError('Version of model to be loaded is unknown')
        model = model_io.load_model(model_class, info_tag, config['checkpoint_version'],
                                    kernel_args={"output_channels": output_channels})
    elif validation:
        raise ValueError("Validation must be performed on model loaded from file.")
    else:
        model: Model = build_model_of(model_class, info_tag,
                                      kernel_args={"output_channels": output_channels})

    if not validation:
        if not config['load_optimizer_from_file']:
            optimizer: torch.optim.Optimizer = build_optimizer(model, config['optimizer'],
                                                               config['optimizer_parameters'])
            model.optimizer = optimizer
    else:
        model.optimizer = None
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

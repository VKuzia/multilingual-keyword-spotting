from typing import Type

import torch.optim
from torch import nn

from config import ArgParser, TrainingConfig, build_optimizer
from models import ModelInfoTag, Model, build_model_of, ModelIOHelper
from src.config.models import get_model_class
from src.dataloaders import MonoMSWCDataset, DataLoaderMode, ClassificationDataLoader, DataLoader
from src.dataloaders.base import SpecDataset, Dataset, TargetProbaFsDataset
from src.models import FewShotModel, CoreModel
from src.transforms.transformers import DefaultTransformer, ValidationTransformer
from trainers.handlers import ModelSaver, ClassificationValidator, Printer, PrinterHandler, \
    ValidationMode
from trainers.trainer import Trainer, DefaultTrainer, TrainingParams
from paths import PATH_TO_SAVED_MODELS, PATH_TO_MSWC_WAV

torch.backends.cudnn.benchmark = True

args = ArgParser().parse_args()
additional_args = {"target_language": None,
                   "target": None,
                   "target_probability": None,
                   "load_embedding": None,
                   "embedding_name": None,
                   "embedding_class": None,
                   "embedding_version": None,
                   "embedding_checkpoint_version": None}
config = TrainingConfig(additional_args).load_json(args.config_path)

for key, value in config:
    print(f'{key}: {value}')

train_non_target_dataset: Dataset = SpecDataset(
    MonoMSWCDataset(PATH_TO_MSWC_WAV, config['target_language'], DataLoaderMode.TRAINING,
                    is_wav=False,
                    predicate=lambda x: x != config['target']),
    DefaultTransformer()
)
train_target_dataset: Dataset = SpecDataset(
    MonoMSWCDataset(PATH_TO_MSWC_WAV, config['target_language'], DataLoaderMode.TRAINING,
                    is_wav=False,
                    predicate=lambda x: x == config['target']),
    DefaultTransformer()
)
train_dataset: Dataset = TargetProbaFsDataset(train_target_dataset, train_non_target_dataset,
                                              config['target'], config['target_probability'])
train_loader: DataLoader = ClassificationDataLoader(train_dataset, config['batch_size'])

validation_non_target_dataset: Dataset = SpecDataset(
    MonoMSWCDataset(PATH_TO_MSWC_WAV, config['target_language'], DataLoaderMode.VALIDATION,
                    is_wav=False,
                    predicate=lambda x: x != config['target']),
    ValidationTransformer()
)
validation_target_dataset: Dataset = SpecDataset(
    MonoMSWCDataset(PATH_TO_MSWC_WAV, config['target_language'], DataLoaderMode.VALIDATION,
                    is_wav=False,
                    predicate=lambda x: x == config['target']),
    ValidationTransformer()
)
validation_dataset: Dataset = TargetProbaFsDataset(validation_target_dataset,
                                                   validation_non_target_dataset, config['target'],
                                                   config['target_probability'])
validation_loader: DataLoader = ClassificationDataLoader(validation_dataset, config['batch_size'])

output_channels = len(train_loader.get_labels())
print(f"output channels: {output_channels}")

info_tag: ModelInfoTag = ModelInfoTag(config['model_name'], config['model_version'])
model_io: ModelIOHelper = ModelIOHelper(PATH_TO_SAVED_MODELS)
model: Model

embedding_class = get_model_class(config['embedding_class'])

model_class: Type[Model] = get_model_class(config['model_class'])
if config['load_model_from_file']:
    if config['checkpoint_version'] is None:
        raise ValueError('Version of model to be loaded is unknown')
    model = model_io.load_model(model_class, info_tag, config['checkpoint_version'],
                                kernel_args={"embedding_class": embedding_class,
                                             "output_channels": 1})
else:
    model: Model = build_model_of(model_class, info_tag,
                                  kernel_args={"embedding_class": embedding_class,
                                               "output_channels": 1})

if config['load_embedding']:

    embedding_info_tag: ModelInfoTag = ModelInfoTag(config['embedding_name'],
                                                    config['embedding_version'])
    if config['embedding_checkpoint_version'] is None:
        raise ValueError('Version of embedding to be loaded is unknown')
    embedding = model_io.load_model(embedding_class, embedding_info_tag,
                                    config['embedding_checkpoint_version'],
                                    kernel_args={"output_channels": 265})
    embedding.kernel.output = nn.Identity()
    for parameter in embedding.kernel.parameters():
        parameter.requires_grad = False
    for param in embedding.kernel.output.parameters():
        param.requires_grad = True
    model.kernel.core = embedding.kernel

if not config['load_optimizer_from_file']:
    optimizer: torch.optim.Optimizer = build_optimizer(model, config['optimizer'],
                                                       config['optimizer_parameters'],
                                                       output_only=True)
    model.optimizer = optimizer

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

trainer.train(model, train_loader, training_params)

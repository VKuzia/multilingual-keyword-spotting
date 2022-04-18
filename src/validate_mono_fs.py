from typing import Type

import torch.optim
from torch import nn
from tqdm import tqdm

from config import ArgParser
from models import ModelInfoTag, Model, build_model_of, ModelIOHelper
from src.config.configs import ValidationConfig
from src.config.models import get_model_class
from src.dataloaders import MonoMSWCDataset, DataLoaderMode, ClassificationDataLoader, DataLoader
from src.dataloaders.base import SpecDataset, Dataset, TargetProbaFsDataset
from src.trainers.handlers.validators import estimate_accuracy_with_errors
from src.transforms.transformers import DefaultTransformer, ValidationTransformer
from paths import PATH_TO_SAVED_MODELS, PATH_TO_MSWC_WAV

torch.backends.cudnn.benchmark = True

args = ArgParser().parse_args()
additional_args = {"language": None,
                   "target_language": None,
                   "target": None,
                   "target_probability": None,
                   "load_embedding": None,
                   "embedding_name": None,
                   "embedding_class": None,
                   "embedding_version": None,
                   "embedding_checkpoint_version": None,
                   "embedding_output_channels": None}
config = ValidationConfig(additional_args).load_json(args.config_path)

for key, value in config:
    print(f'{key}: {value}')

language = config["language"]

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
                                    kernel_args={
                                        "output_channels": config['embedding_output_channels']})
    embedding.kernel.output = nn.Identity()
    for parameter in embedding.kernel.parameters():
        parameter.requires_grad = False
    for param in embedding.kernel.output.parameters():
        param.requires_grad = True
    model.kernel.core = embedding.kernel

model.kernel.eval()
val_accuracy, val_errors = estimate_accuracy_with_errors(model, validation_loader,
                                                         config['batches_per_validation'])

train_accuracy, train_errors = estimate_accuracy_with_errors(model, train_loader,
                                                             config['batches_per_validation'])

val_errors = list(sorted(val_errors.items(), key=lambda item: -item[1]))
train_errors = list(sorted(train_errors.items(), key=lambda item: -item[1]))

val_labels = validation_loader.get_labels()
with open('val_errors.txt', 'w') as output:
    output.write("label -> models output: count\n")
    output.write(f"total count: {config['batches_per_validation'] * config['batch_size']}\n")
    output.write(f"with target probability: {config['target_probability']}\n")
    output.write(f"validation accuracy: {val_accuracy}\n\n")
    for idx, count in tqdm(val_errors):
        output.write(f'{val_labels[idx[1]]} -> {val_labels[idx[0]]}: {count}\n')

train_labels = train_loader.get_labels()
with open('train_errors.txt', 'w') as output:
    output.write("label -> models output: count\n")
    output.write(f"total count: {config['batches_per_validation'] * config['batch_size']}\n")
    output.write(f"with target probability: {config['target_probability']}\n")
    output.write(f"train accuracy: {train_accuracy}\n\n")
    for idx, count in tqdm(train_errors):
        output.write(f'{train_labels[idx[1]]} -> {train_labels[idx[0]]}: {count}\n')

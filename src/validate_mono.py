import torch.optim
from tqdm import tqdm

from config import ArgParser, TrainingConfig
from models import ModelInfoTag, Model, ModelIOHelper
from src.config.models import get_model_class
from src.dataloaders import MonoMSWCDataset, DataLoaderMode, ClassificationDataLoader, DataLoader
from src.trainers.handlers.validators import estimate_accuracy_with_errors
from paths import PATH_TO_SAVED_MODELS, PATH_TO_MSWC_WAV

torch.backends.cudnn.benchmark = True

args = ArgParser().parse_args()
config = TrainingConfig({"language": None, "model_class": None}).load_json(args.config_path)

language = config["language"]
info_tag: ModelInfoTag = ModelInfoTag(config['model_name'], config['model_version'])
model_io: ModelIOHelper = ModelIOHelper(PATH_TO_SAVED_MODELS)
model: Model

train_loader: DataLoader = ClassificationDataLoader(
    MonoMSWCDataset(PATH_TO_MSWC_WAV, language, DataLoaderMode.TRAINING, is_wav=False),
    config['batch_size'])

validation_loader: DataLoader = ClassificationDataLoader(
    MonoMSWCDataset(PATH_TO_MSWC_WAV, language, DataLoaderMode.VALIDATION, is_wav=False),
    config['batch_size'])
output_channels = len(validation_loader.get_labels())
print(output_channels)

model_class = get_model_class(config['model_class'])
if config['load_model_from_file']:
    if config['checkpoint_version'] is None:
        raise ValueError('Version of model to be loaded is unknown')
    model = model_io.load_model(model_class, info_tag, config['checkpoint_version'],
                                kernel_args={"output_channels": output_channels})
else:
    raise ValueError("Validation must be performed on model loaded from file.")

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
    output.write(f"validation accuracy: {val_accuracy}\n\n")
    for idx, count in tqdm(val_errors):
        output.write(f'{val_labels[idx[1]]} -> {val_labels[idx[0]]}: {count}\n')

train_labels = train_loader.get_labels()
with open('train_errors.txt', 'w') as output:
    output.write("label -> models output: count\n")
    output.write(f"total count: {config['batches_per_validation'] * config['batch_size']}\n")
    output.write(f"train accuracy: {train_accuracy}\n\n")
    for idx, count in tqdm(train_errors):
        output.write(f'{train_labels[idx[1]]} -> {train_labels[idx[0]]}: {count}\n')

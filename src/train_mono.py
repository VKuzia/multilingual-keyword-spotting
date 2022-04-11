import torch.optim

from config import ArgParser, TrainingConfig, build_optimizer
from models import ModelInfoTag, Model, build_model_of, ModelIOHelper
from src.config.models import get_model_class
from src.dataloaders import MonoMSWCDataset, DataLoaderMode, ClassificationDataLoader, DataLoader
from src.dataloaders.base import SpecDataset, Dataset
from src.models import FewShotModel, CoreModel
from src.transforms.transformers import DefaultTransformer, ValidationTransformer
from trainers.handlers import ModelSaver, ClassificationValidator, Printer, PrinterHandler, \
    ValidationMode
from trainers.trainer import Trainer, DefaultTrainer, TrainingParams
from paths import PATH_TO_SAVED_MODELS, PATH_TO_MSWC_WAV

torch.backends.cudnn.benchmark = True

args = ArgParser().parse_args()
config = TrainingConfig({"language": None}).load_json(args.config_path)

for key, value in config:
    print(f'{key}: {value}')

language = config["language"]
info_tag: ModelInfoTag = ModelInfoTag(config['model_name'], config['model_version'])
model_io: ModelIOHelper = ModelIOHelper(PATH_TO_SAVED_MODELS)
model: Model

train_dataset: Dataset = SpecDataset(
    MonoMSWCDataset(PATH_TO_MSWC_WAV, language, DataLoaderMode.TRAINING, is_wav=False),
    DefaultTransformer())

train_loader: DataLoader = ClassificationDataLoader(train_dataset, config['batch_size'])

validation_dataset: Dataset = SpecDataset(
    MonoMSWCDataset(PATH_TO_MSWC_WAV, language, DataLoaderMode.VALIDATION, is_wav=False),
    ValidationTransformer()
)

validation_loader: DataLoader = ClassificationDataLoader(validation_dataset, config['batch_size'])

output_channels = len(train_loader.get_labels())
print(output_channels)

model_class = get_model_class(config['model_class'])
if config['load_model_from_file']:
    if config['checkpoint_version'] is None:
        raise ValueError('Version of model to be loaded is unknown')
    model = model_io.load_model(model_class, info_tag, config['checkpoint_version'],
                                kernel_args={"output_channels": output_channels})
else:
    model: Model = build_model_of(model_class, info_tag,
                                  kernel_args={"output_channels": output_channels})

if not config['load_optimizer_from_file']:
    optimizer: torch.optim.Optimizer = build_optimizer(model, config['optimizer'],
                                                       config['optimizer_parameters'])
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

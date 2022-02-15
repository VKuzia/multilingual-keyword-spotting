import torch.optim

from config import ArgParser, TrainingConfig, build_optimizer
from models import ModelInfoTag, Model, build_model_of, ModelIOHelper
from src.dataloaders import MonoMSWCDataset, DataLoaderMode, ClassificationDataLoader, \
    FewShotDataLoader, DataLoader
from src.models import FewShotModel
from trainers.handlers import TimeEpochHandler, StepLossHandler, ModelSaver
from trainers.handlers import ClassificationValidator
from trainers.trainer import Trainer, DefaultTrainer, TrainingParams
from paths import PATH_TO_SAVED_MODELS, PATH_TO_MSWC_WAV

args = ArgParser().parse_args()
config = TrainingConfig({"language": None}).load_json(args.config_path)

language = config["language"]
info_tag: ModelInfoTag = ModelInfoTag(config['model_name'], config['model_version'])
model_io: ModelIOHelper = ModelIOHelper(PATH_TO_SAVED_MODELS)
model: Model

train_loader: DataLoader = \
    FewShotDataLoader(MonoMSWCDataset(PATH_TO_MSWC_WAV, language, DataLoaderMode.TRAINING),
                      MonoMSWCDataset(PATH_TO_MSWC_WAV, language, DataLoaderMode.TRAINING),
                      config['batch_size'], target='dos', target_probability=0.1)
validation_loader: DataLoader = \
    FewShotDataLoader(MonoMSWCDataset(PATH_TO_MSWC_WAV, language, DataLoaderMode.VALIDATION),
                      MonoMSWCDataset(PATH_TO_MSWC_WAV, language, DataLoaderMode.VALIDATION),
                      config['batch_size'], target='dos', target_probability=0.1)

output_channels = len(train_loader.get_labels())

if config['load_model_from_file']:
    if config['checkpoint_version'] is None:
        raise ValueError('Version of model to be loaded is unknown')
    model = model_io.load_model(FewShotModel, info_tag, config['checkpoint_version'],
                                kernel_args={"output_channels": output_channels})
else:
    model: Model = build_model_of(FewShotModel, info_tag,
                                  kernel_args={"output_channels": output_channels})

if not config['load_optimizer_from_file']:
    optimizer: torch.optim.Optimizer = build_optimizer(model, config['optimizer'],
                                                       config['optimizer_parameters'])
    model.optimizer = optimizer

time_measure_handler: TimeEpochHandler = TimeEpochHandler()
validator: ClassificationValidator = \
    ClassificationValidator(validation_loader, batch_count=config['batches_per_validation'])
trainer: Trainer = DefaultTrainer([time_measure_handler], [StepLossHandler()],
                                  [time_measure_handler, validator, ModelSaver(model_io)])

training_params: TrainingParams = TrainingParams(batch_count=config['batches_per_epoch'],
                                                 batch_size=config['batch_size'],
                                                 epoch_count=config['epochs'])

trainer.train(model, train_loader, training_params)

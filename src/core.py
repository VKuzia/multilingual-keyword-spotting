import torch.optim

from src.config.argparser import ArgParser
from src.config.training_config import TrainingConfig
from src.config.optimizers import build_optimizer
from src.models.model import ModelInfoTag, Model, build_model_of
from src.models.model_loader import ModelIOHelper
from src.models.speech_commands.core_dataloader import CoreDataLoader, SpeechCommandsMode
from src.models.speech_commands.core_model import CoreModel2
from src.paths import PATH_TO_SAVED_MODELS, PATH_TO_SPEECH_COMMANDS
from src.trainers.handlers.handlers import TimeEpochHandler, StepLossHandler, ModelSaver
from src.trainers.handlers.validators import ClassificationValidator
from src.trainers.trainer import Trainer, DefaultTrainer, TrainingParams

args = ArgParser().parse_args()
config = TrainingConfig().load_json(args.config_path)

info_tag: ModelInfoTag = ModelInfoTag(config['model_name'], config['model_version'])
model_io: ModelIOHelper = ModelIOHelper(PATH_TO_SAVED_MODELS)
model: Model

if config['load_model_from_file']:
    if config['checkpoint_version'] is None:
        raise ValueError('Version of model to be loaded is unknown')
    model = model_io.load_model(CoreModel2, info_tag, config['checkpoint_version'])
else:
    model: Model = build_model_of(CoreModel2, info_tag)

if not config['load_optimizer_from_file']:
    optimizer: torch.optim.Optimizer = build_optimizer(model, config['optimizer'],
                                                       config['optimizer_parameters'])
    model.optimizer = optimizer

train_loader: CoreDataLoader = \
    CoreDataLoader(PATH_TO_SPEECH_COMMANDS, SpeechCommandsMode.TRAINING, config['batch_size'])
validation_loader: CoreDataLoader = \
    CoreDataLoader(PATH_TO_SPEECH_COMMANDS, SpeechCommandsMode.VALIDATION, config['batch_size'])

time_measure_handler: TimeEpochHandler = TimeEpochHandler()
validator: ClassificationValidator = \
    ClassificationValidator(validation_loader, batch_count=config['batches_per_validation'])
trainer: Trainer = DefaultTrainer([time_measure_handler], [StepLossHandler()],
                                  [time_measure_handler, validator, ModelSaver(model_io)])

training_params: TrainingParams = TrainingParams(batch_count=config['batches_per_epoch'],
                                                 batch_size=config['batch_size'],
                                                 epoch_count=config['epochs'])

trainer.train(model, train_loader, training_params)

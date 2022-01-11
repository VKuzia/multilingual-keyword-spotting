import torch.optim

from config import ArgParser
from config import TrainingConfig
from config import build_optimizer
from models import ModelInfoTag, Model, build_model_of
from models import ModelIOHelper
from models.speech_commands import CoreDataLoader, SpeechCommandsMode
from models.speech_commands.classification import CoreModel2
from trainers.handlers.handlers import TimeEpochHandler, StepLossHandler, ModelSaver
from trainers.handlers.validators import ClassificationValidator
from trainers.trainer import Trainer, DefaultTrainer, TrainingParams
from paths import PATH_TO_SAVED_MODELS, PATH_TO_SPEECH_COMMANDS

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

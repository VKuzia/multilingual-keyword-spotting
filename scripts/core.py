import torch

from models.model import ModelInfoTag, build_model_of, Model
from models.model_loader import ModelIOHelper
from models.speech_commands.core_dataloader import CoreDataLoader
from models.speech_commands.core_model import CoreModel, CoreModel2
from scripts.paths import PATH_TO_SPEECH_COMMANDS, PATH_TO_SAVED_MODELS

from trainers.handlers.handlers import TimeEpochHandler, StepLossHandler, ModelSaver
from trainers.handlers.validators import ClassificationValidator
from trainers.trainer import DefaultTrainer, Trainer, TrainingParams

info_tag: ModelInfoTag = ModelInfoTag("core_embedding", "0_0_2")
model_io: ModelIOHelper = ModelIOHelper(PATH_TO_SAVED_MODELS)
# model: Model = build_model_of(CoreModel2, info_tag)
model: Model = model_io.load_model(CoreModel2, info_tag, 45)
model.optimizer = torch.optim.SGD(model.kernel.parameters(), lr=0.005)

train_loader: CoreDataLoader = CoreDataLoader(PATH_TO_SPEECH_COMMANDS, CoreDataLoader.Mode.TRAINING, 128)
validation_loader: CoreDataLoader = CoreDataLoader(PATH_TO_SPEECH_COMMANDS, CoreDataLoader.Mode.VALIDATION, 128)

time_measure_handler: TimeEpochHandler = TimeEpochHandler()
validator: ClassificationValidator = ClassificationValidator(validation_loader, batch_count=5)
trainer: Trainer = DefaultTrainer([time_measure_handler], [StepLossHandler()],
                                  [time_measure_handler, validator, ModelSaver(model_io)])

training_params: TrainingParams = TrainingParams(batch_count=1000, batch_size=128, epoch_count=50)

trainer.train(model, train_loader, training_params)

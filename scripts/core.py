from models.model import ModelInfoTag, build_model_of, Model
from models.speech_commands.core_dataloader import CoreDataLoader
from models.speech_commands.core_model import CoreModel
from scripts.paths import PATH_TO_SPEECH_COMMANDS

from trainers.handlers.handlers import TimeEpochHandler, ClassificationValidator, StepLossHandler
from trainers.trainer import DefaultTrainer, Trainer, TrainingParams

model: Model = build_model_of(CoreModel, ModelInfoTag("core_embedding", "0_0_1"))

train_loader: CoreDataLoader = CoreDataLoader(PATH_TO_SPEECH_COMMANDS, CoreDataLoader.Mode.TRAINING, 1000)
validation_loader: CoreDataLoader = CoreDataLoader(PATH_TO_SPEECH_COMMANDS, CoreDataLoader.Mode.VALIDATION, 1000)

time_measure_handler: TimeEpochHandler = TimeEpochHandler()
validator: ClassificationValidator = ClassificationValidator(validation_loader, batch_count=5)
trainer: Trainer = DefaultTrainer([time_measure_handler], [StepLossHandler()], [time_measure_handler, validator])

training_params: TrainingParams = TrainingParams(batch_count=1000, batch_size=1000, epoch_count=1)

# Example of training the model via provided abstractions
trainer.train(model, train_loader, training_params)

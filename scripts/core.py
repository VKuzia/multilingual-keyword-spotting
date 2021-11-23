from models.model import ModelInfoTag, build_model_of, Model
from models.model_loader import ModelIOHelper
from models.speech_commands.core_dataloader import CoreDataLoader
from models.speech_commands.core_model import CoreModel
from scripts.paths import PATH_TO_SPEECH_COMMANDS, PATH_TO_SAVED_MODELS

from trainers.handlers.handlers import TimeEpochHandler, ClassificationValidator, StepLossHandler, ModelSaver
from trainers.trainer import DefaultTrainer, Trainer, TrainingParams

model_io: ModelIOHelper = ModelIOHelper(PATH_TO_SAVED_MODELS)
# model: Model = model_io.load_model()
model: Model = build_model_of(CoreModel, ModelInfoTag("core_embedding", "0_0_1"))

train_loader: CoreDataLoader = CoreDataLoader(PATH_TO_SPEECH_COMMANDS, CoreDataLoader.Mode.TRAINING, 128)
validation_loader: CoreDataLoader = CoreDataLoader(PATH_TO_SPEECH_COMMANDS, CoreDataLoader.Mode.VALIDATION, 128)

time_measure_handler: TimeEpochHandler = TimeEpochHandler()
validator: ClassificationValidator = ClassificationValidator(validation_loader, batch_count=5)
trainer: Trainer = DefaultTrainer([time_measure_handler], [StepLossHandler()],
                                  [time_measure_handler, validator, ModelSaver(model_io)])

training_params: TrainingParams = TrainingParams(batch_count=1000, batch_size=128, epoch_count=3)

trainer.train(model, train_loader, training_params)

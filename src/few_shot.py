import torch
from torch import nn

from src.models.few_shot import FewShotSpeechCommandsDataLoader
from src.models.few_shot import FewShotModel, FewShotKernel
from models.model import ModelInfoTag, build_model_of, Model
from models.model_loader import ModelIOHelper
from models.speech_commands.core_dataloader import CoreDataLoader, SpeechCommandsMode
from models.speech_commands.core_model import CoreModel, CoreModel2
from scripts.paths import PATH_TO_SPEECH_COMMANDS, PATH_TO_SAVED_MODELS

from trainers.handlers.handlers import TimeEpochHandler, StepLossHandler, ModelSaver
from trainers.handlers.validators import ClassificationValidator
from trainers.trainer import DefaultTrainer, Trainer, TrainingParams

info_tag: ModelInfoTag = ModelInfoTag("few_shot", "0_0_2")
model_io: ModelIOHelper = ModelIOHelper(PATH_TO_SAVED_MODELS)
# embedding: nn.Module = model_io.load_model(CoreModel2, ModelInfoTag("core_embedding", "0_0_2"), 180).kernel
# model: Model = build_model_of(FewShotModel, info_tag, kernel=FewShotKernel(embedding))
model: Model = model_io.load_model(FewShotModel, info_tag, 39)
model.optimizer = torch.optim.SGD(model.kernel.parameters(), lr=0.00001)

train_loader = FewShotSpeechCommandsDataLoader('happy', PATH_TO_SPEECH_COMMANDS, SpeechCommandsMode.TRAINING, 128, 0.1)
validation_loader = FewShotSpeechCommandsDataLoader('happy', PATH_TO_SPEECH_COMMANDS, SpeechCommandsMode.VALIDATION,
                                                    128, 0.1)

time_measure_handler: TimeEpochHandler = TimeEpochHandler()
validator: ClassificationValidator = ClassificationValidator(validation_loader, batch_count=5)
trainer: Trainer = DefaultTrainer([time_measure_handler], [StepLossHandler()],
                                  [time_measure_handler, validator, ModelSaver(model_io)])

training_params: TrainingParams = TrainingParams(batch_count=1000, batch_size=128, epoch_count=11)

trainer.train(model, train_loader, training_params)

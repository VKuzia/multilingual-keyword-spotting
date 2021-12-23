import typing

from typing import List

from matplotlib import pyplot as plt

from src.models.model import ModelInfoTag, Model
from src.models.model_loader import ModelIOHelper
from src.models.speech_commands.core_dataloader import CoreDataLoader, SpeechCommandsMode
from src.models.speech_commands.core_model import CoreModel, CoreModel2
from src.paths import PATH_TO_SAVED_MODELS, PATH_TO_STATS, PATH_TO_SPEECH_COMMANDS

from trainers.handlers.validators import estimate_accuracy

info_tag: ModelInfoTag = ModelInfoTag("core_embedding", "0_0_2")
model_io: ModelIOHelper = ModelIOHelper(PATH_TO_SAVED_MODELS)

train_loader: CoreDataLoader = CoreDataLoader(PATH_TO_SPEECH_COMMANDS,
                                              SpeechCommandsMode.TRAINING, 128)
validation_loader: CoreDataLoader = CoreDataLoader(PATH_TO_SPEECH_COMMANDS,
                                                   SpeechCommandsMode.VALIDATION,
                                                   128)

epochs_range: typing.Iterable = range(1, 201)
train_accuracies: List[float] = []
validation_accuracies: List[float] = []
batch_count: int = 10

for i in epochs_range:
    print("validating on {} epochs".format(i))
    model: Model = model_io.load_model(CoreModel2, info_tag, i)
    validation_accuracies.append(estimate_accuracy(model, validation_loader, batch_count))
    train_accuracies.append(estimate_accuracy(model, train_loader, batch_count))

figure = plt.figure(figsize=(16, 5))
plt.plot(epochs_range, train_accuracies, label="train accuracy", color="b")
plt.plot(epochs_range, validation_accuracies, label="validation accuracy", color="r")
plt.legend()
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.xticks(epochs_range)
figure.savefig(PATH_TO_STATS + "core_embedding_0_0_2_[180].pdf")
plt.show()

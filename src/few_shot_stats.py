import typing

from typing import List

from matplotlib import pyplot as plt
import numpy as np

from src.models.speech_commands.few_shot import FewShotSpeechCommandsDataLoader
from src.models.speech_commands.few_shot import FewShotModel
from models.model import ModelInfoTag, Model
from models.model_loader import ModelIOHelper
from models.speech_commands import SpeechCommandsMode
from paths import PATH_TO_SPEECH_COMMANDS, PATH_TO_SAVED_MODELS, PATH_TO_STATS

from trainers.handlers import estimate_accuracy

info_tag: ModelInfoTag = ModelInfoTag("few_shot", "0_0_2")
model_io: ModelIOHelper = ModelIOHelper(PATH_TO_SAVED_MODELS)

train_loader = FewShotSpeechCommandsDataLoader('happy', PATH_TO_SPEECH_COMMANDS,
                                               SpeechCommandsMode.TRAINING, 128, 0.1)
validation_loader = FewShotSpeechCommandsDataLoader('happy', PATH_TO_SPEECH_COMMANDS,
                                                    SpeechCommandsMode.VALIDATION,
                                                    128, 0.1)

epochs_range: typing.Iterable = range(1, 51)
train_accuracies: List[float] = []
validation_accuracies: List[float] = []
batch_count: int = 10

for i in epochs_range:
    print("validating on {} epochs".format(i))
    model: Model = model_io.load_model(FewShotModel, info_tag, i)
    validation_accuracies.append(estimate_accuracy(model, validation_loader, batch_count))
    train_accuracies.append(estimate_accuracy(model, train_loader, batch_count))

figure = plt.figure(figsize=(16, 5))
plt.plot(epochs_range, train_accuracies, label="train accuracy", color="b")
plt.plot(epochs_range, validation_accuracies, label="validation accuracy", color="r")
plt.legend()
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.xticks(epochs_range)
figure.savefig(PATH_TO_STATS + "few_shot_0_0_2[20].pdf")
plt.show()

print(np.array(train_accuracies[10:]).mean())
print(np.array(validation_accuracies[10:]).mean())

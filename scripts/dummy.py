"""
This script provides an example of use of all project's abstractions.
It implements a learning of a linear neural network on random batches of floats.

Is used for architecture construction only. Should be deleted soon.
"""

import torch

from dataloaders.dataloader import DataLoader
from models.dummy.dummy_dataloader import DummyDataLoader
from models.dummy.dummy_model import DummyModel
from models.model import ModelInfoTag, build_model_of, Model
from models.model_loader import ModelIOHelper
from scripts.paths import PATH_TO_SAVED_MODELS
from trainers.handlers.handlers import TimeEpochHandler, StepLossHandler, ClassificationValidator
from trainers.trainer import Trainer, TrainingParams, DefaultTrainer

# Initializing the model and training parameters
model: Model = build_model_of(DummyModel, ModelInfoTag("my_dummy_model", "0_0_2"))

data_loader: DataLoader = DummyDataLoader()
validation_data_loader: DataLoader = DummyDataLoader()

time_measure_handler: TimeEpochHandler = TimeEpochHandler()
validator: ClassificationValidator = ClassificationValidator(validation_data_loader, batch_count=10)
trainer: Trainer = DefaultTrainer([time_measure_handler], [StepLossHandler()], [time_measure_handler, validator])

training_params: TrainingParams = TrainingParams(batch_count=5, batch_size=100000, epoch_count=10)

# Example of training the model via provided abstractions
trainer.train(model, data_loader, training_params)

# Example of model's prediction
data = torch.rand((1, 128)).to('cuda')
print(model(data))

# Example of saving model
model_io: ModelIOHelper = ModelIOHelper(PATH_TO_SAVED_MODELS)
model_io.save_model(model)

# Example of loading model
model: Model = model_io.load_model(DummyModel, ModelInfoTag("my_dummy_model", "0_0_2"), 1)
data = torch.rand((1, 128)).to('cuda')
print(model(data))

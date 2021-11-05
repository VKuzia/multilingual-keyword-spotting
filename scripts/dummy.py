"""
This script provides an example of use of all project's abstractions.
It implements a learning of a linear neural network on random batches of floats.

Is used for architecture construction only. Should be deleted soon.
"""

import torch

from dataloaders.dataloader import DataLoader
from models.dummy.dummy_dataloader import DummyDataLoader
from models.dummy.dummy_model import DummyModel
from models.model import ModelInfoTag
from trainers.handlers.handlers import TimeEpochHandler, StepLossHandler, ClassificationValidator
from trainers.trainer import Trainer, TrainingParams, DefaultTrainer

# Initializing the model and training parameters
model: DummyModel = DummyModel(ModelInfoTag("my_dummy_model", "0_0_1"))

data_loader: DataLoader = DummyDataLoader()
validation_data_loader: DataLoader = DummyDataLoader()

time_measure_handler: TimeEpochHandler = TimeEpochHandler()
validator: ClassificationValidator = ClassificationValidator(validation_data_loader, batch_count=10, batch_size=10000)
trainer: Trainer = DefaultTrainer([time_measure_handler], [StepLossHandler()], [time_measure_handler, validator])

training_params: TrainingParams = TrainingParams(batch_count=5, batch_size=100000, epoch_count=10)

# Example of training the model via provided abstractions
trainer.train(model, data_loader, training_params)

# Example of model's prediction
data = torch.rand((1, 128)).to('cuda')
print(model(data))

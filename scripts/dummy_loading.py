import torch

from models.dummy.dummy_model import DummyModel
from models.model import Model, ModelInfoTag
from models.model_loader import ModelIOHelper
from scripts.paths import PATH_TO_SAVED_MODELS

model_io: ModelIOHelper = ModelIOHelper(PATH_TO_SAVED_MODELS)
# model: Model = model_io.load_model_by_path(DummyModel, "my_dummy_model_0_0_1_[1]")
model: Model = model_io.load_model(DummyModel, ModelInfoTag("my_dummy_model", "0_0_1"), 1)

data = torch.rand((1, 128)).to('cuda')
print(model(data))

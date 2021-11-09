import torchvision

from models.core.core_model import CoreModel
from models.model import ModelInfoTag
from scripts.paths import PATH_TO_EFFICIENT_NET

model = CoreModel(PATH_TO_EFFICIENT_NET, ModelInfoTag("core_model", "0_0_1"))
# model = torchvision.models.efficientnet_b0(pretrained=True)
print(model.kernel)


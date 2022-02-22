from src.dataloaders import DataLoader, ClassificationDataLoader, MonoMSWCDataset, DataLoaderMode
from src.dataloaders.base import MultiWalkerDataset
from src.models import ModelIOHelper, ModelInfoTag, CoreModel, Model
from src.paths import PATH_TO_MSWC_WAV, PATH_TO_SAVED_MODELS
from src.trainers.handlers import estimate_accuracy

languages = ['pl']
training_datasets = [MonoMSWCDataset(PATH_TO_MSWC_WAV, language, DataLoaderMode.TRAINING) for
                     language in languages]

validation_datasets = [MonoMSWCDataset(PATH_TO_MSWC_WAV, language, DataLoaderMode.VALIDATION) for
                       language in languages]

training_loader: DataLoader = ClassificationDataLoader(MultiWalkerDataset(training_datasets), 64)
validation_loader: DataLoader = ClassificationDataLoader(MultiWalkerDataset(validation_datasets),
                                                         64)

model_io: ModelIOHelper = ModelIOHelper(PATH_TO_SAVED_MODELS)

info_tag: ModelInfoTag = ModelInfoTag("test_pl", "0_0_0")
model: Model = model_io.load_model(CoreModel, info_tag, 5, kernel_args={
    "output_channels": len(training_loader.get_labels())})

print(estimate_accuracy(model, training_loader, 1000))

print(estimate_accuracy(model, validation_loader, 1000))

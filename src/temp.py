from src.dataloaders import DataLoader, ClassificationDataLoader, MonoMSWCDataset, DataLoaderMode
from src.dataloaders.base import MultiWalkerDataset
from src.paths import PATH_TO_MSWC_WAV

languages = ['en', 'es']
training_datasets = [MonoMSWCDataset(PATH_TO_MSWC_WAV, language, DataLoaderMode.TRAINING) for
                     language in languages]

validation_datasets = [MonoMSWCDataset(PATH_TO_MSWC_WAV, language, DataLoaderMode.VALIDATION) for
                       language in languages]

loader: DataLoader = ClassificationDataLoader(MultiWalkerDataset(training_datasets), 32)

print(loader.get_labels())
print(loader.get_batch())

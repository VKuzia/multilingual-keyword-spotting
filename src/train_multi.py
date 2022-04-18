from typing import List

import torch.optim

from config import ArgParser, TrainingConfig
from src.config import Config
from src.dataloaders import MonoMSWCDataset, DataLoaderMode, ClassificationDataLoader, DataLoader
from src.dataloaders.base import SpecDataset, Dataset, MultiDataset
from src.transforms.transformers import DefaultTransformer
from src.utils.routines import build_default_trainer, build_default_model, get_multi_loader
from paths import PATH_TO_MSWC_WAV


def main():
    torch.backends.cudnn.benchmark = True
    args = ArgParser().parse_args()
    config = TrainingConfig(
        {"languages": None, "model_class": None, "dataset_part": None}).load_json(
        args.config_path)
    for key, value in config:
        print(f'{key}: {value}')

    train_loader: DataLoader = get_multi_loader(config, DataLoaderMode.TRAINING)
    validation_loader: DataLoader = get_multi_loader(config, DataLoaderMode.VALIDATION)
    output_channels = len(train_loader.get_labels())
    print("output_channels:", output_channels)

    model, model_io = build_default_model(config, config['languages'], output_channels)
    trainer, training_params = build_default_trainer(config,
                                                     train_loader,
                                                     validation_loader,
                                                     model_io)
    trainer.train(model, train_loader, training_params)


if __name__ == "__main__":
    main()

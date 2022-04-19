import os

import torch.optim
from tqdm import tqdm

from config import ArgParser
from src.config.configs import ValidationConfig
from src.dataloaders import DataLoaderMode, DataLoader, ClassificationDataLoader
from src.trainers.handlers.validators import estimate_accuracy_with_errors
from src.routines.routines import get_multi_dataset, build_default_validation_model


def estimate_errors(config, model, loader: DataLoader, mode: str, dir: str = "./out_errors"):
    accuracy, errors = estimate_accuracy_with_errors(model, loader,
                                                     config['batches_per_validation'])
    errors = list(sorted(errors.items(), key=lambda item: -item[1]))
    labels = loader.get_labels()
    if not os.path.exists(dir):
        os.makedirs(dir)
    with open(f'{dir}/{model.info_tag.get_name()}_{mode}_errors.txt', 'w') as output:
        output.write("label -> models output: count\n")
        output.write(f"total count: {config['batches_per_validation'] * config['batch_size']}\n")
        output.write(f"{mode} accuracy: {accuracy}\n\n")
        for idx, count in tqdm(errors):
            output.write(f'{labels[idx[1]]} -> {labels[idx[0]]}: {count}\n')


def main():
    torch.backends.cudnn.benchmark = True
    args = ArgParser().parse_args()
    config = ValidationConfig({"languages": None,
                               "dataset_part": None,
                               "model_class": None,
                               "load_model_from_file": True,
                               }).load_json(args.config_path)
    for key, value in config:
        print(f'{key}: {value}')

    train_loader: DataLoader = ClassificationDataLoader(
        get_multi_dataset(config, DataLoaderMode.TRAINING), config['batch_size'])
    validation_loader: DataLoader = ClassificationDataLoader(
        get_multi_dataset(config, DataLoaderMode.VALIDATION), config['batch_size'])
    output_channels = len(train_loader.get_labels())
    print("output_channels:", output_channels)

    model, _ = build_default_validation_model(config, config['languages'],
                                              output_channels)
    model.kernel.eval()
    estimate_errors(config, model, validation_loader, "validation")
    estimate_errors(config, model, train_loader, "train")


if __name__ == "__main__":
    main()

import torch.optim

from config import ArgParser, TrainingConfig
from src.dataloaders import DataLoaderMode, ClassificationDataLoader, DataLoader
from src.utils.routines import build_default_trainer, build_default_model, get_multi_dataset


def main():
    torch.backends.cudnn.benchmark = True
    args = ArgParser().parse_args()
    config = TrainingConfig(
        {"languages": None, "model_class": None, "dataset_part": None}).load_json(
        args.config_path)
    for key, value in config:
        print(f'{key}: {value}')

    train_loader: DataLoader = ClassificationDataLoader(
        get_multi_dataset(config, DataLoaderMode.TRAINING), config['batch_size'])
    validation_loader: DataLoader = ClassificationDataLoader(
        get_multi_dataset(config, DataLoaderMode.VALIDATION), config['batch_size'])
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

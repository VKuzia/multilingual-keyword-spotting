import torch.optim

from config import ArgParser, TrainingConfig
from src.dataloaders import DataLoaderMode, ClassificationDataLoader, DataLoader, MonoMSWCDataset
from src.dataloaders.base import Dataset, TargetProbaFsDataset
from src.paths import PATH_TO_MSWC_WAV
from src.routines.routines import get_multi_dataset, build_default_trainer
from src.routines.routines_fs import build_default_fs_model

torch.backends.cudnn.benchmark = True


def get_dataloader_on_target(config: TrainingConfig, mode: DataLoaderMode) -> DataLoader:
    target_dataset: Dataset = MonoMSWCDataset(PATH_TO_MSWC_WAV, config['target_language'], mode,
                                              is_wav=False, part=config['dataset_part'],
                                              predicate=lambda x: x == config['target'])
    non_target_dataset: Dataset = get_multi_dataset(config, mode, lambda x: x != config['target'])
    dataset: Dataset = TargetProbaFsDataset(target_dataset, non_target_dataset,
                                            config['target'], config['target_probability'])
    return ClassificationDataLoader(dataset, config['batch_size'])


def main():
    args = ArgParser().parse_args()
    additional_args = {"languages": None,
                       "target_language": None,
                       "dataset_part": None,
                       "target": None,
                       "target_probability": None,
                       "load_embedding": None,
                       "embedding_name": None,
                       "embedding_class": None,
                       "embedding_version": None,
                       "embedding_checkpoint_version": None}
    config = TrainingConfig(additional_args).load_json(args.config_path)

    for key, value in config:
        print(f'{key}: {value}')

    train_loader: DataLoader = get_dataloader_on_target(config, DataLoaderMode.TRAINING)
    validation_loader: DataLoader = get_dataloader_on_target(config, DataLoaderMode.VALIDATION)

    model, model_io = build_default_fs_model(config, config['languages'])

    trainer, training_params = build_default_trainer(config,
                                                     train_loader,
                                                     validation_loader,
                                                     model_io)
    trainer.train(model, train_loader, training_params)


if __name__ == "__main__":
    main()

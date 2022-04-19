import torch.optim

from config import ArgParser, TrainingConfig
from src.dataloaders import DataLoaderMode, ClassificationDataLoader, DataLoader, MonoMSWCDataset
from src.dataloaders.base import Dataset, TargetProbaFsDataset
from src.paths import PATH_TO_MSWC_WAV
from src.utils.routines import get_multi_dataset
from src.utils.routines_fs import build_default_fs_model, build_default_fs_validation_model
from src.validate import estimate_errors

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
                       "embedding_class": None,
                       "load_embedding": False}
    config = TrainingConfig(additional_args).load_json(args.config_path)

    for key, value in config:
        print(f'{key}: {value}')

    train_loader: DataLoader = get_dataloader_on_target(config, DataLoaderMode.TRAINING)
    validation_loader: DataLoader = get_dataloader_on_target(config, DataLoaderMode.VALIDATION)

    model, _ = build_default_fs_validation_model(config, config['languages'])
    model.kernel.eval()
    estimate_errors(config, model, validation_loader, "validation")
    estimate_errors(config, model, train_loader, "train")


if __name__ == "__main__":
    main()

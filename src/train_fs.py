import math
import os
from typing import List, Tuple

import torch.optim
from tqdm import tqdm

from config import ArgParser, TrainingConfig
from src.config import Config, get_optimizer_class, get_scheduler_class
from src.dataloaders import DataLoaderMode, ClassificationDataLoader, DataLoader, MonoMSWCDataset
from src.dataloaders.base import Dataset, TargetProbaFsDataset, SampledDataset, SpecDataset, \
    MultiDataset
from src.paths import PATH_TO_MSWC_WAV
from src.routines.routines import build_default_trainer
from src.routines.routines_fs import build_default_fs_model
from src.trainers import DefaultTrainer, Trainer, TrainingParams
from src.trainers.handlers import Printer, PrinterHandler, ClassificationValidator, ValidationMode, \
    ModelSaver, estimate_accuracy
from src.transforms import SpecAugTransformer, ValidationTransformer
from src.utils import rand_indices

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt


def build_loaders(config: Config, target: str, target_dataset, train_non_target_dataset,
                  val_non_target_dataset) -> Tuple[DataLoader, DataLoader]:
    target_indices: List[int] = rand_indices(len(target_dataset) - 1, config['target_count'])
    target_dataset = SampledDataset(target_dataset, target_indices)

    train_target_indices = rand_indices(len(target_dataset) - 1, config['k'])
    train_target_dataset: Dataset = SampledDataset(target_dataset, train_target_indices)

    train_dataset: Dataset = TargetProbaFsDataset(train_target_dataset, train_non_target_dataset,
                                                  target,
                                                  target_proba=config['target_probability'])
    train_loader: DataLoader = ClassificationDataLoader(SpecDataset(train_dataset,
                                                                    SpecAugTransformer()),
                                                        config['batch_size'])

    val_target_indices = list(set(range(len(target_dataset))) - set(train_target_indices))
    val_target_dataset: Dataset = SpecDataset(SampledDataset(target_dataset, val_target_indices),
                                              SpecAugTransformer())

    val_dataset: Dataset = TargetProbaFsDataset(val_target_dataset, val_non_target_dataset,
                                                target, config['target_probability'])
    validation_loader: DataLoader = ClassificationDataLoader(val_dataset, config['batch_size'])

    return train_loader, validation_loader


def plot_history(data, name: str):
    # Fit a normal distribution to the data:
    mu, std = norm.fit(data)
    # Plot the histogram.
    plt.hist(data, bins=50, density=True, alpha=0.6, color='g')
    # Plot the PDF.
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 1000)
    p = norm.pdf(x, mu, std)
    plt.plot(x, p, 'k', linewidth=2)
    title = f"{name}: mu = {mu},  std = {std}"
    print(data.mean())
    plt.title(title)
    plt.show()


def main():
    additional_args = {"languages": None,
                       "target_language": None,
                       "bank_dataset_part": None,
                       "target_dataset_part": None,
                       "val_dataset_part": None,
                       "k": 5,
                       "targets": None,
                       "target_probability": None,
                       "target_count": None,
                       "train_non_target_count": None,
                       "val_non_target_count": None,
                       "tries": None,
                       "load_embedding": None,
                       "embedding_name": None,
                       "embedding_class": None,
                       "embedding_version": None,
                       "embedding_checkpoint_version": None,
                       "seed": 29}
    args = ArgParser().parse_args()
    config = TrainingConfig(additional_args).load_json(args.config_path)

    torch.backends.cudnn.benchmark = True
    for key, value in config:
        print(f'{key}: {value}')

    np.random.seed(config['seed'])

    target_pbar = tqdm(config['targets'])
    for target in target_pbar:
        target_pbar.set_postfix_str(target)
        train_accuracies: List[float] = []
        val_accuracies: List[float] = []

        target_dataset: Dataset = MonoMSWCDataset(PATH_TO_MSWC_WAV,
                                                  config['target_language'],
                                                  DataLoaderMode.ANY,
                                                  is_wav=False,
                                                  part=config['target_dataset_part'],
                                                  predicate=lambda x: x == target)
        train_non_target_datasets: List[Dataset] = []
        for language in sorted(config['languages']):
            base = MonoMSWCDataset(PATH_TO_MSWC_WAV,
                                   language,
                                   DataLoaderMode.ANY,
                                   is_wav=False,
                                   part=config['bank_dataset_part'],
                                   predicate=lambda x: x != target)
            train_non_target_datasets.append(SpecDataset(base, SpecAugTransformer()))
        train_non_target_dataset: Dataset = MultiDataset(train_non_target_datasets)

        train_non_target_indices: List[int] = rand_indices(len(train_non_target_dataset) - 1,
                                                           config['train_non_target_count'])
        train_non_target_dataset = SampledDataset(train_non_target_dataset,
                                                  train_non_target_indices)

        val_non_target_datasets: List[Dataset] = []
        for language in sorted(config['languages']):
            base = MonoMSWCDataset(PATH_TO_MSWC_WAV,
                                   language,
                                   DataLoaderMode.ANY,
                                   is_wav=False,
                                   part=config['val_dataset_part'],
                                   predicate=lambda x: x != target)
            val_non_target_datasets.append(base)
        val_non_target_dataset: Dataset = MultiDataset(val_non_target_datasets)

        val_non_target_indices: List[int] = rand_indices(len(val_non_target_dataset) - 1,
                                                         config['val_non_target_count'])
        val_non_target_dataset = SampledDataset(val_non_target_dataset, val_non_target_indices)

        tries_pbar = tqdm(range(config['tries']), leave=False)
        model, model_io = build_default_fs_model(config, config['languages'],
                                                 dataset_part=config['val_dataset_part'])
        train_loader, validation_loader = build_loaders(config, target, target_dataset,
                                                        train_non_target_dataset,
                                                        val_non_target_dataset)

        printer: Printer = Printer(config['epochs'], config['batches_per_epoch'])
        printer_handler: PrinterHandler = PrinterHandler(printer)
        trainer: Trainer = DefaultTrainer([], [printer_handler], [printer_handler])

        training_params: TrainingParams = TrainingParams(
            batch_count=config['batches_per_epoch'],
            batch_size=config['batch_size'],
            epoch_count=config['epochs'])

        for _ in tries_pbar:

            # reinit the weights
            stdv = 1. / math.sqrt(model.kernel.output[0].weight.size(1))
            model.kernel.output[0].weight.data.uniform_(-stdv, stdv)
            if model.kernel.output[0].bias is not None:
                model.kernel.output[0].bias.data.uniform_(-stdv, stdv)


            optimizer_class = get_optimizer_class(config['optimizer_class'])
            optimizer_params = config['optimizer_parameters']
            scheduler_class = get_scheduler_class(config['scheduler_class'])
            scheduler_params = config['scheduler_parameters']
            optimizer = optimizer_class(model.kernel.output.parameters(), **optimizer_params)
            scheduler = scheduler_class(optimizer, **scheduler_params)

            model.optimizer = optimizer
            model.scheduler = scheduler

            model.kernel.train()
            trainer.train(model, train_loader, training_params)
            model.kernel.eval()
            train_accuracies.append(
                estimate_accuracy(model, train_loader, config['batches_per_validation']))
            val_accuracies.append(
                estimate_accuracy(model, validation_loader, config['batches_per_validation']))
            tries_pbar.set_postfix_str(
                "val_accuracy: {:.6f}, train_accuracy: {:.6f}".format(val_accuracies[-1],
                                                                      train_accuracies[-1]))

        # #####!!!!
        # model_io.save_model(model)

        # plot_history(np.array(train_accuracies), 'Train')
        # plot_history(np.array(val_accuracies), 'Validation')
        # os.mkdir('output')
        with open(f"output/{config['model_name']}_{target}_val.txt", 'w') as output:
            output.write(f"{val_accuracies}\n")

        with open(f"output/{config['model_name']}_{target}_train.txt", 'w') as output:
            output.write(f"{train_accuracies}\n")


if __name__ == "__main__":
    main()

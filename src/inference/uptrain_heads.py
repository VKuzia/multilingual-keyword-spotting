import argparse
import json
import os
from typing import Tuple

from tqdm import tqdm

import src.utils as utils
import src.dataloaders as data
import src.models as models
import src.trainers as trainers
import src.trainers.handlers as handlers
import src.transforms as transforms


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=argparse.FileType('r'))
    parser.add_argument('--datapath', type=utils.dir_path, help='path to dataset dir')
    parser.add_argument('--saved-models', type=utils.dir_path)
    parser.add_argument('--model-output-path', type=utils.dir_path)
    parser.add_argument('--device', default='cuda:0')
    return parser.parse_args()


def get_dataloader(config, datapath: str, keyword: str) -> (data.DataLoader, data.DataLoader):
    lambda_ = lambda word: word == '_negative' or word == keyword
    train_dataset: data.Dataset = data.TableDataset(os.path.join(datapath, config['data']['root']),
                                                    os.path.join(datapath,
                                                                 config['data']['table']),
                                                    data.DataLoaderMode.TRAINING,
                                                    config['data'].get('is_wav', False),
                                                    lambda_)
    train_dataset = data.TransformedDataset(train_dataset, transforms.SpecAugTransformer())
    train_loader: data.DataLoader = data.ClassificationDataLoader(train_dataset,
                                                                  config['data']['batch_size'],
                                                                  config['data'].get('cuda', True))
    return train_loader


def build_trainer(config, train_loader: data.DataLoader, model_io: models.ModelIO,
                  output_dir: str) -> Tuple[trainers.Trainer, trainers.TrainingParams]:
    training_validator: handlers.MetricHandler = \
        handlers.MetricHandler(train_loader,
                               batch_count=config['batches_per_validation'],
                               mode=handlers.ValidationMode.TRAINING,
                               metrics=['xent', 'multiacc'])

    printer: handlers.Printer = handlers.Printer(config['epochs'], config['batches_per_epoch'],
                                                 ['xent_train', 'xent_val', 'multiacc_val'])
    printer_handler: handlers.PrinterHandler = handlers.PrinterHandler(printer)
    saver: handlers.ModelSaver = handlers.ModelSaver(model_io, config, output_dir,
                                                     epoch_rate=config['save_after'])
    post_epoch_handlers = [training_validator, saver, printer_handler]
    trainer: trainers.Trainer = trainers.DefaultTrainer([], [printer_handler], post_epoch_handlers)
    training_params: trainers.TrainingParams = trainers.TrainingParams(
        batch_count=config['batches_per_epoch'],
        batch_size=config['data']['batch_size'],
        epoch_count=config['epochs'])
    return trainer, training_params


def main():
    args = _parse_args()
    config = json.loads(args.config.read())
    model_io: models.ModelIO = models.ModelIO(args.saved_models)
    embedding: models.Module = model_io.build_module(config['model']['embedding'],
                                                     args.saved_models)
    for target in tqdm(config['targets']):
        train_loader = get_dataloader(config, args.datapath, keyword=target)
        trainer, params = build_trainer(config, train_loader, model_io,
                                        os.path.join(args.model_output_path, target))
        head: models.Module = model_io.build_module(config['model']['head_pattern'],
                                                    args.saved_models)
        kernel: models.TotalKernel = models.TotalKernel(embedding, head)
        kernel = kernel.to(args.device)
        optimizer = model_io.build_optimizer(config['optimizer'], kernel, args.saved_models)
        scheduler = model_io.build_scheduler(config['scheduler'], optimizer, args.saved_models)
        loss = model_io.build_loss(config['loss'])
        model = models.Model(kernel, optimizer, scheduler, loss)
        trainer.train([model], train_loader, params)
        model_io.save_model(config, model, os.path.join(args.model_output_path, target, 'final'))
    print('DONE')

if __name__ == "__main__":
    main()

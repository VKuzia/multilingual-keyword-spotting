import argparse
import copy
import json
import os
from typing import List

import torch
from tqdm import tqdm

import src.utils as utils
import src.models as models
import src.dataloaders as data
import src.trainers as trainers
import src.trainers.handlers as handlers


# import warnings
#
# warnings.filterwarnings("ignore")
from src.transforms import transformers


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=argparse.FileType('r'),
                        help='path to train config in *.json')
    parser.add_argument('--datapath', type=utils.dir_path, help='path to dataset dir')
    parser.add_argument('--saved-models', type=utils.dir_path, help='path to saved models dir',
                        default='./saved')
    parser.add_argument('--log', type=str, dest='logdir', help='path to save log to')
    return parser.parse_args()


def get_dataloader(config, datapath, mode) -> data.DataLoader:
    dataset: data.Dataset = data.TableDataset(os.path.join(datapath, config['root']),
                                              os.path.join(datapath, config['table']),
                                              mode,
                                              config.get('is_wav', False))
    if mode == data.DataLoaderMode.TRAINING:
        dataset = data.TransformedDataset(dataset, transformers.DefaultTransformer())
    loader: data.DataLoader = data.ClassificationDataLoader(dataset,
                                                            config[
                                                                'val_batch_size'] if mode == data.DataLoaderMode.VALIDATION else
                                                            config['batch_size'],
                                                            config.get('cuda', True))
    return loader


def main():
    args = _parse_args()
    config = json.loads(args.config.read())
    model_io: models.ModelIO = models.ModelIO(args.saved_models)
    embeddings: List[models.Module] = []
    for model_config in config['models']:
        assert model_config['head']['output'] == 2
        # specific_config = copy.deepcopy(config)
        # specific_config['model'] = model_config
        embedding = model_io.build_module(model_config['embedding'], args.saved_models)
        embeddings.append(embedding)
        # models_list.append(model_io.build_model(specific_config))
    log = []
    if not config.get('datas'):
        if config['data'].get('is_experiment_dir', False):
            config['datas'] = []
            for file in os.listdir(os.path.join(args.datapath, config['data']['path'])):
                data_copy = copy.deepcopy(config['data'])
                # data_copy['path'] = os.path.join(data_copy['path'], file)
                data_copy['table'] = os.path.join(data_copy['path'], file)
                config['datas'].append(data_copy)
    for data_config in tqdm(config['datas'], desc='datas'):
        models_list = []
        for embedding, model_config in zip(embeddings, config['models']):
            head = model_io.build_module(model_config['head'], args.saved_models)

            torch.nn.init.xavier_uniform_(head.linear.weight)
            optimizer = model_io.build_optimizer(config['optimizer'], head, args.saved_models)
            scheduler = model_io.build_scheduler(config['scheduler'], optimizer, args.saved_models)
            kernel = models.TotalKernel(embedding, head)
            loss = model_io.build_loss(config['loss'])
            models_list.append(models.Model(kernel, optimizer, scheduler, loss))
        train_loader: data.DataLoader = get_dataloader(data_config, args.datapath,
                                                       data.DataLoaderMode.TRAINING)
        val_loader: data.DataLoader = get_dataloader(data_config, args.datapath,
                                                     data.DataLoaderMode.VALIDATION)
        val_validator: handlers.MetricHandler = \
            handlers.MetricHandler(val_loader,
                                   batch_count=None,
                                   mode=handlers.ValidationMode.VALIDATION,
                                   metrics=['roc_auc', 'binary_f1_score', 'eer', 'precision', 'recall'])
        train_resetter = handlers.DataloaderResetter(train_loader)
        val_resetter = handlers.DataloaderResetter(val_loader)
        # printer: handlers.Printer = handlers.Printer(config['epochs'], train_loader.get_batch_count(), ['roc_auc', 'eer'])
        # printer_handler = handlers.PrinterHandler(printer)

        trainer: trainers.DefaultTrainer = trainers.DefaultTrainer([train_resetter, val_resetter],
                                                                   [], [])
        training_params: trainers.TrainingParams = trainers.TrainingParams(
            batch_count=None,
            batch_size=config['datas'][0]['batch_size'],
            epoch_count=config['epochs'])
        trainer.train(models_list, train_loader, training_params)
        for model in models_list:
            val_validator.handle(model)
        model_metrics = [x.learning_info.to_dict()['metrics'] for x in models_list]
        log.append({"data": data_config['table'], "metrics": model_metrics})
    with open(os.path.join(args.logdir, 'log.json'), 'w') as log_output:
        json.dump(log, log_output, indent=2)
    with open(os.path.join(args.logdir, 'log_config.json'), 'w') as config_output:
        if not config.get('datas'):
            del config['datas']
        json.dump(config, config_output, indent=2)

    print('DONE')


if __name__ == "__main__":
    main()

import argparse
import json
import os

import src.utils as utils
import src.dataloaders as data
import src.models as models
import src.trainers as trainers
import src.trainers.handlers as handlers


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=argparse.FileType('r'),
                        help='path to train config in *.json')
    parser.add_argument('--datapath', type=utils.dir_path, help='path to dataset dir')
    parser.add_argument('--saved-models', type=utils.dir_path, help='path to saved models dir',
                        default='./saved')
    parser.add_argument('--output', type=str, help='path to checkpoint dir')
    return parser.parse_args()


def get_dataloaders(config, datapath: str) -> (data.DataLoader, data.DataLoader):
    train_dataset: data.Dataset = data.TableDataset(os.path.join(datapath, config['data']['root']),
                                                    os.path.join(datapath,
                                                                  config['data']['table']),
                                                    data.DataLoaderMode.TRAINING,
                                                    config['data'].get('from_spectrograms', False))
    train_loader: data.DataLoader = data.ClassificationDataLoader(train_dataset,
                                                                  config['data']['batch_size'],
                                                                  config['data'].get('cuda', True))

    val_dataset: data.Dataset = \
        data.TableDataset(os.path.join(datapath, config['data']['root']),
                          os.path.join(datapath,
                                        config['data'].get('val_table', config['data']['table'])),
                          data.DataLoaderMode.VALIDATION,
                          config['data'].get('is_wav', False))
    val_loader: data.DataLoader = \
        data.ClassificationDataLoader(val_dataset,
                                      config['data'].get('val_batch_size',
                                                         config['data']['batch_size']),
                                      config['data'].get('cuda', True))
    return train_loader, val_loader


def build_trainer(config, train_loader: data.DataLoader,
                  val_loader: data.DataLoader, model_io: models.ModelIO,
                  output_dir: str) -> trainers.Trainer:
    validation_validator: handlers.MetricHandler = \
        handlers.MetricHandler(val_loader,
                               batch_count=config['batches_per_validation'],
                               mode=handlers.ValidationMode.VALIDATION,
                               metrics=['xent', 'multiacc'])
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
    post_epoch_handlers = [validation_validator, training_validator, saver, printer_handler]
    trainer: trainers.Trainer = trainers.DefaultTrainer([], [printer_handler], post_epoch_handlers)
    return trainer


def main():
    args = _parse_args()
    config = json.loads(args.config.read())
    train_loader, val_loader = get_dataloaders(config, args.datapath)
    assert len(train_loader.get_labels()) == config['model']['head']['output']
    assert len(val_loader.get_labels()) == config['model']['head']['output']
    model_io: models.ModelIO = models.ModelIO(args.saved_models)
    model: models.Model = model_io.build_model(config)
    trainer: trainers.Trainer = build_trainer(config, train_loader, val_loader, model_io,
                                              args.output)
    training_params: trainers.TrainingParams = trainers.TrainingParams(
        batch_count=config['batches_per_epoch'],
        batch_size=config['data']['batch_size'],
        epoch_count=config['epochs'])
    trainer.train(model, train_loader, training_params)
    model_io.save_model(config, model, os.path.join(args.output, 'final'))
    print('DONE')


if __name__ == "__main__":
    main()

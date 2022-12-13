import argparse
import json
import os
from typing import List

import numpy as np
from matplotlib import pyplot as plt

import src.utils as utils
import src.dataloaders as data
import src.trainers.handlers as handlers
import src.models as models


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=argparse.FileType('r'),
                        help='config including data and model architecture in *.json')
    parser.add_argument('--datapath', type=utils.dir_path, help='path to dataset dir')
    parser.add_argument('--saved-models', type=utils.dir_path, help='path to saved models dir',
                        default='./saved')
    parser.add_argument('--list-part', dest='part', type=int, help='amount of labels to be shown')
    parser.add_argument('--output', type=str, help='path to save most frequent errors')
    return parser.parse_args()


def get_dataloader(config, datapath: str) -> data.DataLoader:
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
    return val_loader


def plot_confusion_matrix(confusion_matrix: np.ndarray, labels: List[str]):
    fig, ax = plt.subplots(figsize=(16, 9))
    im = ax.imshow(confusion_matrix)

    ax.set_xticks(np.arange(len(labels)), labels=labels)
    ax.set_yticks(np.arange(len(labels)), labels=labels)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(labels)):
        for j in range(len(labels)):
            text = ax.text(j, i, confusion_matrix[i, j],
                           ha="center", va="center", color="w")

    ax.set_xlabel("prediction")
    ax.set_ylabel("target")
    ax.set_title("Confusion matrix")
    fig.tight_layout()
    plt.show()


def get_most_common_errors(preds, labels_tensor, labels):
    counts = {}
    for pred, l in zip(preds, labels_tensor):
        if pred.item() != l.item():
            tup = (labels[pred.item()], labels[l.item()])
            if not counts.get(tup):
                counts[tup] = 0
            counts[tup] += 1
    return sorted(list(counts.items()), key=lambda x: -x[1])


def main():
    args = _parse_args()
    config = json.loads(args.config.read())
    val_loader = get_dataloader(config, args.datapath)
    assert len(val_loader.get_labels()) == config['model']['head']['output']
    model_io: models.ModelIO = models.ModelIO(args.saved_models)
    embedding: models.Module = model_io.build_module(config['model']['embedding'],
                                                     args.saved_models)
    head: models.Module = model_io.build_module(config['model']['head'],
                                                args.saved_models)
    model: models.Module = models.TotalKernel(embedding, head).cuda()
    outputs, preds, labels = utils.run_through(model, val_loader)
    # part_labels = val_loader.get_labels()[:args.part] if args.part else val_loader.get_labels()
    # confusion_matrix = utils.build_confusion_matrix(preds.cpu(), labels.cpu())
    # plot_confusion_matrix(confusion_matrix, part_labels)
    error_list = get_most_common_errors(preds, labels, val_loader.get_labels())
    accuracy = handlers.estimate_multiclass_accuracy(labels, preds)
    f1_score = handlers.estimate_weighted_f1_score(labels, preds)

    errors_sum = sum((x[1] for x in error_list))

    with open(args.output, 'w') as output:
        output.write(f'total_samples: {len(preds)}\n')
        output.write(f'wrong_answers: {errors_sum}\n')
        output.write(f'val_f1_score: {f1_score}\n')
        output.write(f'val_accuracy: {accuracy}\n')
        for key, value in error_list:
            output.write(f'Model gave "{key[0]}" on label "{key[1]}": {value}\n')

    print('DONE')


if __name__ == "__main__":
    main()

import argparse
import os

import numpy as np
import pandas as pd
from tqdm import tqdm


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, required=True)
    parser.add_argument('--languages', nargs='+', required=True)
    parser.add_argument('--labels-to-leave', type=int, default=10)
    parser.add_argument('--output-dir', type=str, required=True)
    return parser.parse_args()


def main():
    args = _parse_args()
    data = pd.read_csv(args.csv, delimiter=',')
    train_dfs = []
    val_dfs = []
    for language in tqdm(args.languages):
        df = data[data['language'] == language]
        labels = list(data['label'].unique())
        val_labels = set(np.random.choice(labels, size=args.labels_to_leave, replace=False))
        train_dfs.append(df[~df['label'].isin(val_labels)])
        val_dfs.append(df[df['label'].isin(val_labels)])
    val_dfs.append(data[~data['language'].isin(args.languages)])
    train = pd.concat(train_dfs, ignore_index=True)[['mode', 'label', 'path', 'language', 'gender']].sample(frac=1)
    val = pd.concat(val_dfs, ignore_index=True)[['mode', 'label', 'path', 'language', 'gender']].sample(frac=1)
    train.to_csv(os.path.join(args.output_dir, 'TRAIN.csv'))
    val.to_csv(os.path.join(args.output_dir, 'VAL.csv'))





if __name__ == "__main__":
    main()

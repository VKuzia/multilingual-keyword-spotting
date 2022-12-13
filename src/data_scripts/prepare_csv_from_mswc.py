import argparse
import os

import pandas as pd
from tqdm import tqdm

import src.utils.csv_tools as tools
import src.utils.helpers as utils


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dirs', type=utils.dir_path, help='path all language csvs')
    parser.add_argument('--top', default=200, type=int, help='Count of popular labels to filter.')
    parser.add_argument('--count', default=140000, type=int,
                        help='Count of total samples to include')
    parser.add_argument('--output', type=str, help='Path to save output.')
    parser.add_argument('--languages', nargs='*', help='List of languages to process.')
    return parser.parse_args()


def main():
    args = _parse_args()
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    for language in tqdm(args.languages):
        path_to_table = os.path.join(args.dirs, f'{language}_splits.csv')
        data = pd.read_csv(path_to_table, delimiter=',').sample(frac=1)
        data = tools.simplify_dataset(data)
        data['path'] = data['path'].apply(lambda x: f'{language}/clips/{x}')
        stats = tools.stats_dataset(data)
        labels = tools.select_by_len_and_top(stats, 3, args.top)
        result = data[data['label'].isin(labels)]
        result = result.reset_index().drop(columns=['index'], axis=1)
        if len(result) > args.count:
            result = result.sample(n=args.count)
        else:
            result = result.sample(frac=1)
        path_to_save = os.path.join(args.output, f'{language}_top_{args.top}_{len(result)}.csv')
        result.reset_index(drop=True).to_csv(path_to_save, index_label='index')
        print(f'[{language.upper()}] train samples: {len(result[result["mode"] == "train"])}')
        print(f'[{language.upper()}] val samples: {len(result[result["mode"] == "val"])}')
        print(f'[{language.upper()}] total samples: {len(result)}')
        print('===============================')


if __name__ == "__main__":
    main()

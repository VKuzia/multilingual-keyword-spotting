import argparse
import os

import pandas as pd

from src import utils


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=utils.dir_path, required=True)
    parser.add_argument('--output', type=str, required=True)
    return parser.parse_args()


def main():
    args = _parse_args()
    dfs = []
    for name in os.listdir(args.dir):
        df = pd.read_csv(os.path.join(args.dir, name), delimiter=',')
        dfs.append(df)
    result = pd.concat(dfs, ignore_index=True, sort=False).sample(frac=1)
    print(f'unique labels: {len(result["label"].unique())}')
    print(f'total len: {len(result)}')
    result.to_csv(args.output)


if __name__ == "__main__":
    main()

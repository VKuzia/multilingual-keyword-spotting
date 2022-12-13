import argparse
import os
import shutil

import pandas as pd
from tqdm import tqdm

from src import utils


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, help='path all language csvs')
    parser.add_argument('--input', type=str)
    parser.add_argument('--output', type=str)
    return parser.parse_args()


def main():
    args = _parse_args()
    paths = pd.read_csv(args.csv, delimiter=',')['path']
    for path_input in tqdm(paths):
        path = path_input.replace('.wav', '.opus')
        dst = os.path.join(args.output, path)
        if not os.path.exists(dst):
            src = os.path.join(args.input, path)
            os.makedirs(dst, exist_ok=True)
            shutil.copy(src, dst)



if __name__ == "__main__":
    main()
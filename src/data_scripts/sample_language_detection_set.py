import argparse
import os

import numpy as np
import pandas as pd
from tqdm import tqdm


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, required=True)
    parser.add_argument('--output-csv', type=str, required=True)
    return parser.parse_args()


def main():
    args = _parse_args()
    data = pd.read_csv(args.csv, delimiter=',')
    data['label'] = data['language']
    data.to_csv(args.output_csv)


if __name__ == "__main__":
    main()

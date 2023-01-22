import argparse
import os
import shutil

from tqdm import tqdm

from src import utils


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dirs', type=utils.dir_path, help='path all language csvs')
    parser.add_argument('--output', type=str)
    return parser.parse_args()


def main():
    args = _parse_args()
    for split_name in tqdm(os.listdir(args.dirs)):
        shutil.copy(os.path.join(args.dirs, split_name, f'{split_name}.csv'),
                    os.path.join(args.output, f'{split_name}.csv'))


if __name__ == "__main__":
    main()

"""
kindly taken from
https://colab.research.google.com/github/harvard-edge/multilingual_kws/blob/main/multilingual_kws_intro_tutorial.ipynb
"""
import argparse
import os
import subprocess
from multiprocessing import Pool
from pathlib import Path

from tqdm import tqdm

from src import utils


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=utils.dir_path, required=True)
    parser.add_argument('--languages', nargs='+', required=True)
    parser.add_argument('--dst', type=utils.dir_path, required=True)
    parser.add_argument('--threads', default=6)
    return parser.parse_args()


# NOTE: opus-tools package is needed!
# install it with: apt-get -qq install opus-tools sox
def handle_sample(sample_dest_tuple):
    sample, dest_file = sample_dest_tuple
    cmd = ["opusdec", "--rate", "16000", "--quiet", sample, dest_file]
    subprocess.run(cmd, stdout=open(os.devnull, 'wb'))


def main():
    args = _parse_args()
    for language in tqdm(args.languages):
        words = sorted(os.listdir(os.path.join(args.src, language, "clips")))
        pbar = tqdm(words)  # for viewing progress
        max_len = max([len(word) for word in words])
        for word in pbar:
            pbar.set_description(f"Converting {language} [{word}{' ' * (max_len - len(word))}]")
            destdir = os.path.join(args.dst, language, "clips", word)
            Path(destdir).mkdir(parents=True, exist_ok=True)
            samples = list(Path(os.path.join(args.src, language, "clips", word)).glob("*.opus"))
            # samples.sort()
            with Pool(args.threads) as pool:
                pool_args = [(sample, os.path.join(destdir, str((sample.stem + ".wav")))) for sample in samples]
                for _ in pool.imap_unordered(handle_sample, pool_args):
                    pass


if __name__ == '__main__':
    main()

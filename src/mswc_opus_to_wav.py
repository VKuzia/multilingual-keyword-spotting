"""
kindly taken from
https://colab.research.google.com/github/harvard-edge/multilingual_kws/blob/main/multilingual_kws_intro_tutorial.ipynb
"""

import os
import subprocess
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import tqdm

from src.paths import PATH_TO_MSWC_OPUS, PATH_TO_MSWC_WAV

CONVERT_ALL = True
N_SAMPLES_TO_CONVERT = 20  # unused if CONVERT_ALL == True
LANGUAGES_LIST = ["ru"]
THREADS_COUNT = 4

SRC = Path(PATH_TO_MSWC_OPUS)
DEST = Path(PATH_TO_MSWC_WAV)

rng = np.random.RandomState(0)  # we also sort FS listings to aid reproducibility


# NOTE: opus-tools package is needed!
# install it with: apt-get -qq install opus-tools sox

def handle_sample(sample_dest_tuple):
    """Converts given file from opus format to wav with constant bitrate of 16000 Hz"""
    sample, dest_file = sample_dest_tuple
    cmd = ["opusdec", "--rate", "16000", "--quiet", sample, dest_file]
    subprocess.run(cmd, stdout=open(os.devnull, 'wb'))


def main():
    for language in LANGUAGES_LIST:
        words = sorted(os.listdir(SRC / language / "clips"))
        pbar = tqdm.tqdm(words)  # for viewing progress
        max_len = max([len(word) for word in words])
        for word in pbar:
            pbar.set_description(f"Converting {language} [{word}{' ' * (max_len - len(word))}]")
            destdir = DEST / language / "clips" / word
            destdir.mkdir(parents=True, exist_ok=True)
            samples = list((SRC / language / "clips" / word).glob("*.opus"))
            samples.sort()
            if not CONVERT_ALL and len(samples) > N_SAMPLES_TO_CONVERT:
                samples = rng.choice(samples, N_SAMPLES_TO_CONVERT, replace=False)
            with Pool(THREADS_COUNT) as pool:
                args = [(sample, destdir / (sample.stem + ".wav")) for sample in samples]
                for _ in pool.imap_unordered(handle_sample, args):
                    pass


if __name__ == '__main__':
    main()

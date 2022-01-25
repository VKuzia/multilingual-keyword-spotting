"""
kindly taken from
https://colab.research.google.com/github/harvard-edge/multilingual_kws/blob/main/multilingual_kws_intro_tutorial.ipynb
"""

import os
import subprocess
from pathlib import Path

import numpy as np
import tqdm

from src.paths import PATH_TO_MSWC, PATH_TO_MSWC_WAV

CONVERT_ALL = True
N_SAMPLES_TO_CONVERT = 20
LANGUAGES_LIST = ["en", "es"]

src = Path(PATH_TO_MSWC)
dest = Path(PATH_TO_MSWC_WAV)
rng = np.random.RandomState(0)  # we also sort FS listings to aid reproducibility


# NOTE: opus-tools package is needed!
# install it with: apt-get -qq install opus-tools sox

def handle_sample(sample, dest_file):
    cmd = ["opusdec", "--rate", "16000", "--quiet", sample, dest_file]
    subprocess.run(cmd, stdout=open(os.devnull, 'wb'))


def main():
    for language in LANGUAGES_LIST:
        words = sorted(os.listdir(src / language / "clips"))
        pbar = tqdm.tqdm(words)  # for viewing progress
        max_len = max([len(word) for word in words])
        for word in pbar:
            pbar.set_description(f"Converting {language} [{word}{' ' * (max_len - len(word))}]")
            destdir = dest / language / "clips" / word
            destdir.mkdir(parents=True, exist_ok=True)
            samples = list((src / language / "clips" / word).glob("*.opus"))
            samples.sort()
            if not CONVERT_ALL and len(samples) > N_SAMPLES_TO_CONVERT:
                samples = rng.choice(samples, N_SAMPLES_TO_CONVERT, replace=False)
            for sample in tqdm.tqdm(samples, leave=False):
                handle_sample(sample, destdir / (sample.stem + ".wav"))


if __name__ == '__main__':
    main()

from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.paths import PATH_TO_MSWC_WAV, PATH_TO_MSWC

LANGUAGES_LIST = ["en", "es"]
CSV_LIST = ["test.csv", "train.csv"]

src = Path(PATH_TO_MSWC)
dest = Path(PATH_TO_MSWC_WAV)


def handle_csv(source, destination):
    data = pd.read_csv(source)['LINK']
    with open(destination, "w") as output:
        for entry in data:
            output.write(entry.replace(".opus", ".wav") + '\n')


for language in LANGUAGES_LIST:
    src_dir = src / language
    dest_dir = dest / language
    pbar = tqdm(CSV_LIST, desc=f"[{language}]")
    for csv_suffix in pbar:
        handle_csv(src_dir / f"{language}_{csv_suffix}",
                   dest_dir / f"{language}_{csv_suffix.replace('.csv', '.txt')}")

import numpy as np
import pandas as pd
from functools import partial

from src.paths import PATH_TO_MSWC_CSV, PATH_TO_MSWC_WAV
from src.utils.csv_tools import select_by_len_and_top, select_by_top, sample, simplify_dataset, stats_dataset, identity

ALL_LANGUAGES = ["cs", "it", "pl", "ru", "tt", "uk"]
LANGUAGES = ['it']
DATASET_PART_NAME = "bank_200+_50000"
# SELECTION_RULE = partial(select_most_frequent, count=180)
# LABEL_SELECTION_RULE = partial(select_by_len_and_top, min_len=4, count=45)
LABEL_SELECTION_RULE = partial(select_by_top, lower=200, upper=None)
SAMPLE_RULE = partial(sample, count=50000)
# SAMPLE_RULE = identity

PATH_TO_CSV = PATH_TO_MSWC_CSV
PATH_TO_OUTPUT = PATH_TO_MSWC_WAV


def main():
    for language in LANGUAGES:
        path_to_csv = f'{PATH_TO_CSV}/{language}/{language}_splits.csv'

        data = pd.read_csv(path_to_csv, delimiter=',')
        data = simplify_dataset(data)
        stats = stats_dataset(data)
        labels = LABEL_SELECTION_RULE(stats)
        result = data[data['label'].isin(labels)]
        result = SAMPLE_RULE(result)
        result = result.reset_index().drop(columns=['index'], axis=1)
        print(
            f"[{language}] label ratio: {len(result['label'].unique()) / len(data['label'].unique())} "
            f"summary ratio: {len(result) / len(data)} "
            f"total: {len(result)}")
        result.to_csv(f'{PATH_TO_OUTPUT}/{language}/{language}_{DATASET_PART_NAME}.csv', ',')


if __name__ == "__main__":
    main()

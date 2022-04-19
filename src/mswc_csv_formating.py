import pandas as pd
from functools import partial

from src.paths import PATH_TO_MSWC_CSV, PATH_TO_MSWC_WAV


def simplify_dataset(data):
    df = data[data['VALID'] == True][['SET', 'LINK', 'WORD']]
    df = df.rename(columns={'SET': 'mode', 'LINK': 'path', 'WORD': 'label'})
    df['mode'] = df['mode'].map({'TEST': 'test', 'DEV': 'val', 'TRAIN': 'train'})
    df['path'] = df['path'].apply(lambda x: x.replace('.opus', '.wav'))
    df = df[['mode', 'label', 'path']]
    return df


def stats_dataset(data):
    df = pd.DataFrame(data['label'].value_counts()).rename(columns={'label': 'count'})
    for mode in data['mode'].unique():
        df[mode] = data[data['mode'] == mode]['label'].value_counts()
        df[mode] = df[mode].fillna(0).astype(dtype=int)
    return df


def select_labels_by_val(data, lower_bound=10):
    return list(data[data['val'] >= lower_bound].index)


def select_most_frequent(data, count):
    df = data.sort_values('count', axis=0, ascending=False)
    return list(df[:count].index)


LANGUAGES = ["cs", "el", "tt", "uk", "ru", "pl"]
DATASET_PART_NAME = "popular_100"

SELECTION_RULE = partial(select_most_frequent, count=100)

PATH_TO_CSV = PATH_TO_MSWC_CSV
PATH_TO_OUTPUT = PATH_TO_MSWC_WAV


def main():
    for language in LANGUAGES:
        path_to_csv = f'{PATH_TO_CSV}/{language}/{language}_splits.csv'

        data = pd.read_csv(path_to_csv, ',')
        data = simplify_dataset(data)
        stats = stats_dataset(data)
        labels = SELECTION_RULE(stats)
        result = data[data['label'].isin(labels)]
        result = result.reset_index().drop(columns=['index'], axis=1)
        print(f"label ratio: {len(result['label'].unique())}/{len(data['label'].unique())} "
              f"summary ratio: {len(result) / len(data)}")
        result.to_csv(f'{PATH_TO_OUTPUT}/{language}/{language}_{DATASET_PART_NAME}.csv', ',')


if __name__ == "__main__":
    main()

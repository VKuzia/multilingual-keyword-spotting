import numpy as np
import pandas as pd


def simplify_dataset(data):
    """
    Translates MSWC csv file into a more convinient form for me :)
    TODO: describe the format
    """
    df = data[data['VALID'] == True][['SET', 'LINK', 'WORD']]
    df = df.rename(columns={'SET': 'mode', 'LINK': 'path', 'WORD': 'label'})
    df['mode'] = df['mode'].map({'TEST': 'test', 'DEV': 'val', 'TRAIN': 'train'})
    df['path'] = df['path'].apply(lambda x: x.replace('.opus', '.wav'))
    df = df[['mode', 'label', 'path']]
    return df


def stats_dataset(data):
    """
    TODO: describe the format
    """
    df = pd.DataFrame(data['label'].value_counts()).rename(columns={'label': 'count'})
    for mode in data['mode'].unique():
        df[mode] = data[data['mode'] == mode]['label'].value_counts()
        df[mode] = df[mode].fillna(0).astype(dtype=int)
    return df


def select_labels_by_val(data, lower_bound=10):
    """Leaves only labels which present in validation subset at least lower_bound times"""
    return list(data[data['val'] >= lower_bound].index)


def select_by_top(data, *, upper=None, lower=0):
    """Leaves count labels with highest 'count' feature"""
    df = data.sort_values('count', axis=0, ascending=False)
    if upper:
        return list(df[lower:upper + 1].index)
    else:
        return list(df[lower:].index)


def select_by_len_and_top(data, min_len, count):
    df = data[data.index.map(lambda x: len(x) >= min_len)]
    df = df.sort_values('count', axis=0, ascending=False)
    return list(df[:count].index)


def sample(data, count):
    indices = np.random.random_integers(low=0, high=count - 1, size=(count,))
    return data.iloc[indices]


def sample_labels(data, count):
    labels = data['label']
    max_index = labels.nunique() - 1
    if count > max_index + 1:
        raise ValueError(f"There are only {max_index + 1} labels, can't sample {count} of them")
    indices = np.random.random_integers(low=0, high=max_index, size=(count,))
    sampled_labels = labels.unique().iloc[indices]
    return data[data['label'].isin(sampled_labels)]


def identity(data):
    return data

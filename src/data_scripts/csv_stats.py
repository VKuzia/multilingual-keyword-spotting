import argparse

import pandas as pd
from matplotlib import pyplot as plt


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, required=True)
    parser.add_argument('--language-key', dest='lang', default='language')
    parser.add_argument('--label-key', dest='label', default='label')
    parser.add_argument('--target', type=str)
    return parser.parse_args()


def plot_labels(all):
    plt.figure(figsize=(16, 9))
    plt.bar([x[0] for x in all], [x[2] for x in all])
    plt.title('languages by labels count')
    plt.show()

def plot_gender(all):
    plt.figure(figsize=(16, 9))
    plt.bar([x[0] for x in all], [x[3] for x in all], color='blue')
    plt.bar([x[0] for x in all], [x[4] for x in all], color='pink')
    plt.title('languages by genders')
    plt.show()


def plot_sum(all):
    plt.figure(figsize=(16, 9))
    plt.bar([x[0] for x in all], [x[1] for x in all])
    plt.title('languages by samples count')
    plt.show()


def main():
    args = _parse_args()
    data = pd.read_csv(args.csv, delimiter=',')
    counts = data[args.lang].value_counts()
    male_counts = data[data['gender'] == 'm'][args.lang].value_counts()
    female_counts = data[data['gender'] == 'f'][args.lang].value_counts()
    languages = list(counts.index)
    all = []
    for language in languages:
        all.append((language, counts[language],
                    len(data[data[args.lang] == language][args.label].unique()),
                    male_counts[language], female_counts[language]))
    print(f'ALL: {len(data)}')
    print('----------------')
    all = sorted(all, key=lambda x: -x[1])
    for lang, sum_, labels, _, _ in all:
        print(f'{lang}: \n{labels} labels; \n{sum_} samples')
        print('------------------')
    plot_sum(all)
    plot_labels(all)
    plot_gender(all)
    if args.target:
        target_data = data[data[args.label] == args.target]
        print('TARGET:')
        print(f'languages: {list(target_data[args.lang].unique())};')
        print(f'count: {len(target_data)};')


if __name__ == "__main__":
    main()

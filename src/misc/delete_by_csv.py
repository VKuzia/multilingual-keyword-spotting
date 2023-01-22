import argparse
import os
import shutil

import pandas as pd
from tqdm import tqdm

from src import utils


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, help='path all language csvs')
    parser.add_argument('--dir', type=str)
    return parser.parse_args()

def main():
    args = _parse_args()
    data = pd.read_csv(args.csv, delimiter=',')

    pairs = {(x, y) for x, y in zip(data['label'], data['language'])}
    paths = set(data['path'])
    assert len(paths) == len(data)
    # for path in tqdm(paths):
    #     new_path = insert_clips_dir(path)
    #     full_path = os.path.join(args.dir, new_path.replace('.wav', '.opus'))
    #     if not os.path.exists(full_path):
    #         print(f'{full_path} does not exist')
    #         return
    # print("ALL exist!")
    # path_files = {x.split('/')[-1] for x in paths}
    # assert len(path_files) == len(data)
    # print(list(path_files)[:10])
    # labels = {x.split('/')[2] for x in paths}
    # print(len(paths))
    # print(len(labels))
    # print(len(pairs))
    # dirs_count = 0
    dirs_to_delete = []
    samples_to_delete = []
    # delete_count = 0
    # all_count = 0
    remained_list = []
    for root, dirs, files in os.walk(args.dir):
        if dirs:
            continue
        split = root.split('/')
        label, lang = split[-1], split[-3]
        if (label, lang) not in pairs:
            dirs_to_delete.append(root)
            # samples_to_delete.extend(os.path.join(lang, label, x) for x in files)
            continue
        for file_input in files:
            file = file_input.replace('.opus', '.wav')
            dir_ = '/'.join(root.split('/')[-3:])
            # print(dir_)
            if os.path.join(dir_, file) in paths:
                remained_list.append(os.path.join(dir_, file_input))
            else:
                samples_to_delete.append(os.path.join(root, file_input))
    print(len(dirs_to_delete))
    print(len(samples_to_delete))
    print(len(data), len(remained_list))
    # print(dirs_to_delete[:10])
    # print(samples_to_delete[:10])
    # x = input()
    # for dir_to_delete in tqdm(dirs_to_delete):
    #     shutil.rmtree(dir_to_delete)
    # for sample in tqdm(samples_to_delete):
    #     os.remove(sample)


        # all_count += len(files)
        # dirs_count += 1
        # split = root.split('/')
        # label, lang = split[-1], split[-3]
        # if (label, lang) not in pairs:
        #     dirs_to_delete.append(root)
        #     delete_count += len(files)
        #     continue
        # if len(files) > 1000:
        #     # print(files[:2])
        #     for file_input in files:
        #         file = file_input.replace('.opus', '.wav')
        #         if file not in path_files:
        #             samples_to_delete.append(os.path.join(root, file_input))
        #             delete_count += 1
        #         else:
        #             remained_list.append(file)
        # else:
        #     remained_list.extend([x.replace('.opus', '.wav') for x in files])

    # print(dirs_count)
    # print(len(dirs_to_delete))
    # print(dirs_count - len(dirs_to_delete))
    # print(dirs_to_delete[:10])
    # print(len(samples_to_delete))
    # print(samples_to_delete[:10])
    # print(all_count)
    # print(delete_count)
    # remained = all_count - delete_count
    # print(remained)
    # print(remained == len(data))
    # remained = [root for root, dirs, _ in os.walk(args.dir) if
    #              not dirs and root.split('/')[-1] in labels]
    # print(len(remained))
    # print(remained[:10])
    # print(list(paths)[:10])
    # counter = 0
    # for root, dirs, files in os.walk(args.dir):
    #     counter += 1
    #     print(root)
    #     print(dirs)
    #     print(files)
    #     if counter >= 30:
    #         break
    # for path_input in tqdm(paths):
    #     path = path_input.replace('.wav', '.opus')
    #     dst = os.path.join(args.output, path)
    #     if not os.path.exists(dst):
    #         src = os.path.join(args.input, path)
    #         os.makedirs(dst, exist_ok=True)
    #         shutil.copy(src, dst)


if __name__ == "__main__":
    main()

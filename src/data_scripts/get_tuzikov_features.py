import argparse
import os
from multiprocessing import Pool
from typing import List

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src import utils


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir-in', type=utils.dir_path, help='path to dir with languages')
    parser.add_argument('--dir-out', type=utils.dir_path, help='output features dir')
    parser.add_argument('--n', type=int, help='dimensionality of tuzikov features')
    parser.add_argument('--x-len', type=int, default=40)
    parser.add_argument('--y-len', type=int, default=49)
    parser.add_argument('--threads', type=int, default=6)
    return parser.parse_args()


def build_moment_matrices(x_len: int, y_len: int, n: int, cuda: bool = True):
    result = [[None for _ in range(n)] for _ in range(n)]
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            temp = torch.zeros(size=(y_len, x_len))
            for y in range(1, y_len + 1):
                for x in range(1, x_len + 1):
                    temp[y - 1][x - 1] = (y ** i) * (x ** j)
            if cuda:
                result[i - 1][j - 1] = temp.cuda()
            else:
                result[i - 1][j - 1] = temp
    return result


def get_tuzikov(matrix: torch.Tensor, moment_matrices: List[List[torch.Tensor]], n: int):
    # center_x = 0
    # center_y = 0
    # for y in range(matrix.shape[0]):
    #     for x in range(matrix.shape[1]):
    #         center_x += matrix[y][x] * x
    #         center_y += matrix[y][x] * y
    result = torch.zeros(size=(n, n))
    for i in range(n):
        for j in range(n):
            result[i][j] = torch.sum(moment_matrices[i][j] * matrix)
    return result / (matrix.shape[0] * matrix.shape[1])


def handle_sample(args):
    path_in, n, moment_matrices, path_out = args
    # fbank = torch.load(path_in, map_location=torch.device('cpu'))
    fbank = torch.load(path_in)
    fbank_min = torch.min(fbank)
    fbank = (fbank - fbank_min) / (torch.max(fbank) - fbank_min)
    tuzikov = get_tuzikov(fbank, moment_matrices, n)
    torch.save(tuzikov, path_out)


def main():
    args = _parse_args()
    moment_matrices = build_moment_matrices(args.x_len, args.y_len, args.n)
    for language in tqdm(os.listdir(args.dir_in)):
        src = os.path.join(args.dir_in, language, 'clips')
        for label in tqdm(os.listdir(src), leave=False):
            label_path = os.path.join(language, 'clips', label)
            os.makedirs(os.path.join(args.dir_out, label_path), exist_ok=True)
            pool_args: List = []
            for record in os.listdir(os.path.join(args.dir_in, label_path)):
                pool_args.append((os.path.join(args.dir_in, label_path, record), args.n,
                                  moment_matrices,
                                  os.path.join(args.dir_out, label_path, record)))
            if args.threads > 1:
                with Pool(args.threads) as pool:
                    for _ in pool.imap_unordered(handle_sample, pool_args):
                        pass
            else:
                for tup in pool_args:
                    handle_sample(tup)



if __name__ == "__main__":
    main()

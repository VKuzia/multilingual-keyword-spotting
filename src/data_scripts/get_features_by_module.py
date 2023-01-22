import argparse
import json
import os
from multiprocessing import Pool
from typing import List

import torch
from tqdm import tqdm

from src import utils, models
import src.dataloaders as data
import src.models as models


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir-in', type=utils.dir_path, help='path to dir with languages')
    parser.add_argument('--dir-out', type=utils.dir_path, help='output features dir')
    parser.add_argument('--config', type=argparse.FileType('r'))
    parser.add_argument('--saved-models', type=utils.dir_path)
    return parser.parse_args()


def main():
    args = _parse_args()
    config = json.loads(args.config.read())
    model_io: models.ModelIO = models.ModelIO(args.saved_models)
    module: models.Module = model_io.build_module(config['model'], args.saved_models).cuda()
    module.eval()
    with torch.no_grad():
        for language in tqdm(os.listdir(args.dir_in)):
            for word in tqdm(os.listdir(os.path.join(args.dir_in, language, 'clips')), leave=False):
                dir_path = os.path.join(language, 'clips', word)
                os.makedirs(os.path.join(args.dir_out, dir_path), exist_ok=True)
                for sample in os.listdir(os.path.join(args.dir_in, language, 'clips', word)):
                    input_tensor = torch.load(os.path.join(args.dir_in, dir_path, sample)) \
                        .unsqueeze(dim=0).unsqueeze(dim=0).transpose(2, 3)
                    output = module.forward(input_tensor).squeeze()
                    torch.save(output, os.path.join(args.dir_out, dir_path, sample))


if __name__ == "__main__":
    main()

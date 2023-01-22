import argparse
import os
import torchaudio
import torch
from tqdm import tqdm

from src import utils


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=utils.dir_path, required=True)
    parser.add_argument('--languages', nargs='+', required=True)
    parser.add_argument('--mels', default=49)
    parser.add_argument('--timestamps', default=40)
    parser.add_argument('--power', default=1.0)
    parser.add_argument('--dst', type=utils.dir_path, required=True)
    return parser.parse_args()


def process_audio(src: str, dest: str, transform, shape) -> None:
    """Loads wav provided by src, transforms it to cuda tensor and saves to dest"""
    waveform, _ = torchaudio.load(src)
    target = torch.zeros(shape)
    transformed = transform(waveform)[0]
    time_shape = min(transformed.shape[1], shape[1])
    target[:, :time_shape] = transformed[:, :time_shape]
    torch.save(target.cuda(), dest)


def main():
    args = _parse_args()
    shape = (args.mels, args.timestamps)
    transform = torchaudio.transforms.MelSpectrogram(n_mels=shape[0],
                                                     hop_length=(16000 + shape[1] - 1) // shape[1],
                                                     power=1)
    for language in tqdm(args.languages):
        path_to_clips = os.path.join(args.src, language, "clips")
        path_to_output = os.path.join(args.dst, language, "clips")

        pbar = tqdm(os.listdir(path_to_clips), leave=False)
        for label in pbar:
            pbar.set_description(label)
            os.makedirs(os.path.join(path_to_output, label), exist_ok=True)
            for audio in os.listdir(os.path.join(path_to_clips, label)):
                process_audio(os.path.join(path_to_clips, label, audio),
                              os.path.join(path_to_output, label, audio).replace(".wav", ".pt"),
                              transform, shape)


if __name__ == "__main__":
    main()

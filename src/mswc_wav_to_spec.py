import os
import torchaudio
import torch
from tqdm import tqdm

LANGUAGES = ["it"]
TRANSFORM = torchaudio.transforms.MelSpectrogram(n_mels=49, hop_length=401, power=1)
# TRANSFORM = torchaudio.transforms.Spectrogram(n_fft=97, hop_length=401)
SPEC_SHAPE = (49, 40)


def process_audio(src: str, dest: str) -> None:
    """Loads wav provided by src, transforms it to cuda tensor and saves to dest"""
    waveform, _ = torchaudio.load(src)
    target = torch.zeros(*SPEC_SHAPE)
    transformed = TRANSFORM(waveform)[0]
    time_shape = min(transformed.shape[1], SPEC_SHAPE[1])
    target[:, :time_shape] = transformed[:, :time_shape]
    torch.save(target.cuda(), dest)


def main():
    for language in LANGUAGES:
        path_to_clips = f"dataset/multilingual_spoken_words/wav/{language}/clips"
        path_to_output = f"dataset/multilingual_spoken_words/wav/{language}/clips_tensors_temp"

        pbar = tqdm(os.listdir(path_to_clips), leave=False)
        for label in pbar:
            pbar.set_description(label)
            os.makedirs(f'{path_to_output}/{label}', exist_ok=True)
            for audio in os.listdir(path_to_clips + "/" + label):
                process_audio(f'{path_to_clips}/{label}/{audio}',
                              f'{path_to_output}/{label}/{audio}'.replace(".wav", ".pt"))


if __name__ == "__main__":
    main()

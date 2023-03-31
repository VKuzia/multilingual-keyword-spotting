import torch
import torchaudio


def get_transformer(config):
    if config['type'] == 'mel_spectrogram':
        return MelSpecTransformer(config)
    else:
        raise ValueError(f'Unknown transformer "{config["type"]}"')


class MelSpecTransformer:

    def __init__(self, config):
        power = config.get('power', 1.0)
        self.n_mels = config['n_mels']
        self.sample_rate = config.get('sample_rate', 16000)
        self.timestamps = config['timestamps']
        self.hop_length = self.sample_rate // self.timestamps
        self.transform = torchaudio.transforms.MelSpectrogram(n_mels=self.n_mels,
                                                              hop_length=self.hop_length + 1,
                                                              power=power)

    def __call__(self, x):
        x = (x - x.mean()) / (x.max() - x.min())
        spec = self.transform(torch.Tensor(x))
        formatted = torch.transpose(spec.reshape(1, 1, self.n_mels, self.timestamps), 2, 3)
        return formatted

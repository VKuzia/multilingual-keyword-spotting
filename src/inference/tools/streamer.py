import struct
from typing import Any, Tuple

import numpy as np
import pyaudio as pyaudio


class AudioStreamer:

    def __init__(self, config):
        audio_format, audio_format_multiplier = self.get_pydub_format(config['format'])
        self.format = audio_format
        self.format_multiplier = audio_format_multiplier
        self.sample_rate = config['sample_rate']
        self.chunks_on_frame = config['chunks_on_frame']
        assert self.sample_rate % self.chunks_on_frame == 0
        self.chunk_size = self.sample_rate // self.chunks_on_frame
        self.channels = config['channels']
        self.refresh_rate_sec = self.chunk_size / self.sample_rate
        self.buffer = self._get_zero_buffer()

    def _get_zero_buffer(self):
        return bytes([0 for _ in range(self.sample_rate * self.format_multiplier)])

    def on_trigger(self):
        self._get_zero_buffer()

    def stream(self):
        p = pyaudio.PyAudio()
        stream = p.open(
            format=self.format,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            output=False,
            frames_per_buffer=self.chunk_size,
        )
        while True:
            data = stream.read(self.chunk_size)
            self.buffer = self.buffer[self.chunk_size * self.format_multiplier:] + data
            unpacked = struct.unpack(str(self.chunk_size * self.chunks_on_frame) + "f", self.buffer)
            np_data = np.array(unpacked)
            yield np_data

    @staticmethod
    def get_pydub_format(format_str) -> Tuple[Any, int]:
        if format_str == 'float_32':
            return pyaudio.paFloat32, 4
        else:
            raise ValueError(f'Unknown format "{format_str}"')

import re
import json
import scipy.io.wavfile
from collections import OrderedDict, Counter


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def load_config(config_path):
    config = AttrDict()
    with open(config_path, "r") as f:
        input_str = f.read()
    input_str = re.sub(r'\\\n', '', input_str)
    input_str = re.sub(r'//.*\n', '\n', input_str)
    data = json.loads(input_str)
    config.update(data)
    return config


def count_parameters(model):
    r"""Count number of trainable parameters in a network"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_sample(file_path, sampling_rate, audio):
    """Helper function to save sample

    Args:
        file_path (str or pathlib.Path): save file path
        sampling_rate (int): sampling rate of audio (usually 22050)
        audio (torch.FloatTensor): torch array containing audio in [-1, 1]
    """
    audio = (audio.numpy() * 32768).astype("int16")
    scipy.io.wavfile.write(file_path, sampling_rate, audio)

from typing import List

import numpy as np
import soundfile
import torch
import torch.nn as nn
from pathlib import Path

from css.training.train import TrainCfg, get_model
from utils.conf import load_yaml_to_dataclass
from utils.mic_array_model import multichannel_mic_pos_xyz_cm


def load_css_model(model_dir: Path) -> (nn.Module, TrainCfg):
    """Load multi-channel (mc) or single-channel (sc) CSS model from checkpoint and yaml files."""

    def fetch_one_file(path: Path, suffix: str):
        files = list(path.glob(suffix))
        if len(files) == 0:
            raise FileNotFoundError(f'expecting at least one {suffix} file in {path}')
        assert len(files) == 1, f'expecting exactly one {suffix} file in {path}'
        return str(files[0])

    yaml_path = fetch_one_file(model_dir, '*.yaml')
    checkpoint_path = fetch_one_file(model_dir, '*.pt')

    train_cfg = load_yaml_to_dataclass(yaml_path, TrainCfg)
    separator = get_model(train_cfg)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    def get_sub_state_dict(state_dict, prefix):
        # during training, model is wrapped in DP/DDP which introduces "module." prefix. remove it.
        return {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}

    separator.load_state_dict(get_sub_state_dict(checkpoint["model"], "module."))
    return separator, train_cfg


def load_audio(wav_file_names: List, is_mc: bool) -> (np.ndarray, int):
    """Loads audio data from wav files and returns it as a numpy array.
    Args:
        wav_file_names: list of wav file names.
        is_mc: True if multi-channel, False if single-channel.
    Returns:
        mix_wav: input audio data [Batch, n_samples, n_channels].
        sr: sample rate.
    """

    dtype = 'float32'
    if is_mc:
        num_mics = len(multichannel_mic_pos_xyz_cm())
        assert len(wav_file_names) == num_mics, f'expecting {num_mics} microphones'
        # Read audio data and sampling rates from all files
        audio_data, srs = zip(*[soundfile.read(wav_file, dtype=dtype) for wav_file in wav_file_names])
        mix_wav = np.stack(audio_data, axis=-1)[np.newaxis, ...]  # -> [Batch, n_samples, n_channels]
        assert mix_wav.ndim == 3 and mix_wav.shape[2] in (1, 7)
        sr = srs[0]
    else:
        assert len(wav_file_names) == 1
        mix_wav, sr = soundfile.read(wav_file_names[0], dtype=dtype)
        assert mix_wav.ndim == 1
        mix_wav = mix_wav[np.newaxis, :, np.newaxis]  # [Batch, n_samples, n_channels]

    return mix_wav, sr

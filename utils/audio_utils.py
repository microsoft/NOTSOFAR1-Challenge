import os
import numpy as np
import soundfile as sf
import scipy.io.wavfile as wf

MAX_INT16 = np.iinfo(np.int16).max
EPSILON = np.finfo(np.float32).eps


def read_wav(fname, beg=None, end=None, normalize=True, return_rate=False):
    """
    Read wave files using scipy.io.wavfile(support multi-channel)
    """
    # samps_int16: N x C or N
    #   N: number of samples
    #   C: number of channels
    if beg is not None:
        samps_int16, samp_rate = sf.read(fname,
                                         start=beg,
                                         stop=end,
                                         dtype="int16")
    else:
        samp_rate, samps_int16 = wf.read(fname)
    # N x C => C x N
    samps = samps_int16.astype(np.float32)
    # tranpose because I used to put channel axis first
    if samps.ndim != 1:
        samps = np.transpose(samps)
    # normalize like MATLAB and librosa
    if normalize:
        samps = samps / MAX_INT16
    if return_rate:
        return samp_rate, samps
    return samps


def write_wav(fname, samps: np.ndarray, sr=16000, max_norm: bool = True):
    """
    Write wav to file

    max_norm: normalize to [-1, 1] to avoid potential overflow.
    """
    assert samps.ndim == 1
    if max_norm:
        samps = samps * 0.99 / (np.max(np.abs(samps)) + 1e-7)

    dir_name = os.path.dirname(fname)
    os.makedirs(dir_name, exist_ok=True)
    sf.write(fname, samps, sr)


def play_wav(wav: np.ndarray, fs: int = 16000, volume_factor: float = 1.):
    import sounddevice as sd
    numpy_audio = wav.squeeze()
    sd.play(numpy_audio * volume_factor, fs)
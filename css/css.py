from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import os
import soundfile
import torch
from scipy.io.wavfile import write
from tqdm import trange

from css.css_with_conformer.separate import Separator
from utils.mic_array_model import multichannel_mic_pos_xyz_cm


# CSS inference configuration
@dataclass
class CssCfg:
    segment_size: float = 3.  # in seconds
    hop_size: float = 1.5     # in seconds
    normalize_segment_power: bool = False
    device: Optional[str] = None
    show_progressbar: bool = True
    checkpoint_sc: str = 'CSS_with_Conformer/sc/1ch_conformer_base'
    checkpoint_mc: str = 'CSS_with_Conformer/mc/conformer_base'
    device_id: int = 0
    num_spks: int = 2  # the number of streams the separation models outputs
    pass_through_ch0: bool = False


def css_inference(out_dir: str, models_dir: str, session: pd.Series, cfg: CssCfg,
                  fetch_from_cache: bool) -> pd.Series:
    """
    Applies CSS to each session in sessions_df.
    Args:
        out_dir: the separated wav files will be saved to out_dir/{module_name}/{session_id}.
        models_dir: directory with CSS models.
            example: project_root/artifacts/css_models/
        session: test sesssion row.
        cfg: CSS configuration.
        fetch_from_cache: If True, returns the cached results if they exist. Otherwise, runs the inference.

    Returns:
        session: the input session with the following added columns:
            sep_wav_file_names: a list of separated file paths (typically 2-3 files)

    """
    print("Running CSS (Continuous Speech Separation)")

    session_css = session.copy()

    assert isinstance(session.wav_file_names, list)
    if cfg.pass_through_ch0:
        session_css['sep_wav_file_names'] = session.wav_file_names[0:1]
        return session_css
    else:
        assert False, 'work in progress'


    css_out_dir = Path(out_dir) /  "css_inference" / session.session_id
    if fetch_from_cache and css_out_dir.exists():
        sep_wav_file_names = sorted(css_out_dir.glob('*.wav'))
        session_css['sep_wav_file_names'] = sep_wav_file_names
        return session_css

    chkpt = str(Path(models_dir) / (cfg.checkpoint_mc if session.is_mc else cfg.checkpoint_sc))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # css-with-conformer model
    separator = Separator(chkpt, get_mask=False, device=device)
    separator.executor.eval()
    dtype ='float32'

    if session.is_mc:
        num_mics = len(multichannel_mic_pos_xyz_cm())
        assert len(session.wav_file_names) == num_mics, f'expecting {num_mics} microphones'
        # Read audio data and sampling rates from all files
        audio_data, srs = zip(*[soundfile.read(wav_file, dtype=dtype) for wav_file in session.wav_file_names])
        mixwav = np.stack(audio_data, axis=-1)  # -> [n_samples, n_channels]
        assert mixwav.ndim == 2 and mixwav.shape[1] in (1, 7)
        sr = srs[0]
    else:
        assert len(session.wav_file_names) == 1

        mixwav, sr = soundfile.read(os.path.join(session.meeting_id,session.wav_file_names[0]), dtype=dtype)
        mixwav = mixwav[16000 * 10:16000 * 20]  # TODO: TEMP!!

        # import sounddevice as sd
        # sd.play(mixwav.squeeze() * 1, 16000)


    waves = separate_and_stitch(mixwav[np.newaxis, ...], separator, sr, cfg)

    out_dir_css = os.path.join(out_dir,'css_inference',session.meeting_id, session.device_name)
    if not os.path.exists(out_dir_css):
        os.makedirs(out_dir_css)

    waves_separated = [] 
    basename = os.path.basename(session.wav_file_names[0])[:-4]
    for i in range(len(waves)):
        filename = os.path.join(out_dir_css,f"{basename}_spk_{i}.wav")
        write(filename, sr, waves[i].squeeze()) 
        print(f"Saved file: {filename}") 
        waves_separated.append(filename)

    # session_css['sep_wav_file_names'] = ...

    return session_css


@torch.no_grad()
def separate_and_stitch(speech_mix: np.ndarray, separator: Separator, fs: int, cfg: CssCfg):
    """ TODO doc
    This code is inspired in part by the SeparateSpeech class from the ESPnet toolkit.

    Args:
        speech_mix: Input speech data (Batch, Nsamples [, Channels])
        fs: sample rate
    Returns:
        [separated_audio1, separated_audio2, ...]

    """
    assert False, 'work in progress'

    assert speech_mix.ndim > 1
    batch_size, mix_length = speech_mix.shape[:2]

    assert mix_length > cfg.segment_size * fs, 'mixture should be at least one segment long'

    overlap_length = int(np.round(fs * (cfg.segment_size - cfg.hop_size)))
    num_segments = int(
        np.ceil((mix_length - overlap_length) / (cfg.hop_size * fs))
    )
    t = T = int(cfg.segment_size * fs)
    pad_shape = speech_mix[:, :T].shape  # [B, T]
    enh_waves = []

    range_ = trange if cfg.show_progressbar else range
    for i in range_(num_segments):
        st = int(i * cfg.hop_size * fs)
        en = st + T
        if en >= mix_length:
            # en - st < T (last segment)
            en = mix_length
            speech_seg = np.zeros_like(speech_mix, shape=pad_shape)
            t = en - st
            speech_seg[:, :t] = speech_mix[:, st:en]
        else:
            t = T
            speech_seg = speech_mix[:, st:en]  # [B, T, {C}]

        # segment-level separation
        assert batch_size, 'assuming 1 example in batch. easy to support more.'
        egs = {'mix': speech_seg.squeeze(0)}
        processed_wav = separator.separate(egs)  # list with (num_spks + num_noise) elements

        sep_wavs = processed_wav[:separator.executor.nnet.num_spks]

        if speech_seg.dim() > 2:
            # multi-channel speech
            ref_channel = 0
            speech_seg_ = speech_seg[:, ref_channel]
        else:
            speech_seg_ = speech_seg

        if cfg.normalize_segment_power:
            # normalize the scale to match the input mixture scale
            mix_energy = torch.sqrt(
                torch.mean(speech_seg_[:, :t].pow(2), dim=1, keepdim=True)
            )
            enh_energy = torch.sqrt(
                torch.mean(
                    sum(processed_wav)[:, :t].pow(2), dim=1, keepdim=True
                )
            )
            processed_wav = [
                w * (mix_energy / enh_energy) for w in processed_wav
            ]
        # List[torch.Tensor(num_spk, B, T)]
        enh_waves.append(torch.stack(processed_wav, dim=0))

    # stitch the enhanced segments together
    waves = enh_waves[0]
    for i in range(1, num_segments):
        # TODO: re-use PIT loss
        # permutation between separated streams in last and current segments
        # perm = self.cal_permumation(
        #     waves[:, :, -overlap_length:],
        #     enh_waves[i][:, :, :overlap_length],
        #     criterion="si_snr",
        # )
        # repermute separated streams in current segment
        for batch in range(batch_size):
            enh_waves[i][:, batch] = enh_waves[i][perm[batch], batch]

        if i == num_segments - 1:
            enh_waves[i][:, :, t:] = 0
            enh_waves_res_i = enh_waves[i][:, :, overlap_length:t]
        else:
            enh_waves_res_i = enh_waves[i][:, :, overlap_length:]

        # overlap-and-add (average over the overlapped part)
        waves[:, :, -overlap_length:] = (
                                                waves[:, :, -overlap_length:] + enh_waves[i][:, :,
                                                                                :overlap_length]
                                        ) / 2
        # concatenate the residual parts of the later segment
        waves = torch.cat([waves, enh_waves_res_i], dim=2)
    # ensure the stitched length is same as input
    assert waves.size(2) == speech_mix.size(1), (waves.shape, speech_mix.shape)
    waves = torch.unbind(waves, dim=0)




    assert len(waves) == cfg.num_spks
    assert len(waves[0]) == batch_size

    # TODO: normalize to max<1 before write
    waves = [w.cpu().numpy() for w in waves]

    return waves
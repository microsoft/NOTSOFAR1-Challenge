from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import os
import soundfile
import torch
import torch.nn.functional as F
import torch as th
from scipy.io.wavfile import write
from torch import nn
from tqdm import trange

from css.css_with_conformer.utils.mvdr_util import make_mvdr
from css.training.conformer_wrapper import ConformerCssWrapper
from css.training.train import TrainCfg, get_model
from css.training.losses import mse_loss, l1_loss, PitWrapper
from utils.audio_utils import write_wav
from utils.logging_def import get_logger
from utils.mic_array_model import multichannel_mic_pos_xyz_cm
from utils.conf import load_yaml_to_dataclass
from utils.numpy_utils import dilate, erode
from utils.plot_utils import plot_stitched_masks, plot_left_right_stitch, plot_separation_methods
from utils.audio_utils import play_wav

_LOG = get_logger('css')


# CSS inference configuration
@dataclass
class CssCfg:
    segment_size_sec: float = 3.  # in seconds
    hop_size_sec: float = 1.5     # in seconds
    normalize_segment_power: bool = False
    stitching_loss: str = 'l1'  # loss function for stitching adjacent segments ('l1' or 'mse')
    stitching_input: str = 'mask' # type of input for stitching loss ('mask' or 'separation_result')
    seg_weight_m0_sec: float = 0.15  # see calc_segment_weight
    seg_weight_m1_sec: float = 0.3
    activity_th: float = 0.4  # threshold for segmentation mask
    activity_dilation_sec: float = 0.4  # dilation and erosion for segmentation mask
    activity_erosion_sec: float = 0.2
    device: Optional[str] = None
    show_progressbar: bool = True
    # segment-wise single-channel model
    checkpoint_sc: str = 'notsofar/conformer1.0/sc'
    # segment-wise multi-channel model
    checkpoint_mc: str = 'notsofar/conformer1.0/mc'
    device_id: int = 0
    num_spks: int = 3  # the number of streams the separation models outputs
    mc_mvdr: bool = True  # if True, applies MVDR to the multi-channel input
    mc_mask_floor_db: float = 0.  # mask floor in db. -inf means no floor. 0 means mask has no effect
    sc_mask_floor_db: float = -np.inf
    pass_through_ch0: bool = False  # if True, simply returns the first channel of the input and skips CSS
    slice_audio_for_debug: bool = False  # if True, only processes 10 seconds of the input audio


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
    _LOG.info("Running CSS (Continuous Speech Separation)")

    session_css = session.copy()

    assert isinstance(session.wav_file_names, list)
    if cfg.pass_through_ch0:
        session_css['sep_wav_file_names'] = session.wav_file_names[0:1]
        return session_css


    css_out_dir = Path(out_dir) /  "css_inference" / session.session_id
    if fetch_from_cache and css_out_dir.exists():
        sep_wav_file_names = sorted(css_out_dir.glob('sep*.wav'))
        session_css['sep_wav_file_names'] = sep_wav_file_names
        return session_css

    # get css-with-conformer model
    separator = load_separator_model(cfg, session.is_mc, models_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    separator.eval()
    dtype ='float32'

    if session.is_mc:
        num_mics = len(multichannel_mic_pos_xyz_cm())
        assert len(session.wav_file_names) == num_mics, f'expecting {num_mics} microphones'
        # Read audio data and sampling rates from all files
        audio_data, srs = zip(*[soundfile.read(wav_file, dtype=dtype) for wav_file in session.wav_file_names])
        mixwav = np.stack(audio_data, axis=-1)[np.newaxis, ...]  # -> [Batch, n_samples, n_channels]
        assert mixwav.ndim == 3 and mixwav.shape[2] in (1, 7)
        sr = srs[0]
    else:
        assert len(session.wav_file_names) == 1
        mixwav, sr = soundfile.read(session.wav_file_names[0], dtype=dtype)
        assert mixwav.ndim == 1
        mixwav = mixwav[np.newaxis, :, np.newaxis]  # [Batch, Nsamples, Channels]

    if cfg.slice_audio_for_debug:
        mixwav = mixwav[:, sr*100:sr*110, :]

    separated_wavs = separate_and_stitch(mixwav, separator, sr, device, cfg)

    write_wav(css_out_dir / 'input_mixture.wav', samps=mixwav[0,:,0], sr=sr)

    sep_wav_file_names = []
    for i, w in enumerate(separated_wavs):
        filename = css_out_dir / f"sep_stream{i}.wav"
        _LOG.info(f"CSS: saving separated wav to {filename}")
        write_wav(filename, samps=w, sr=sr)
        sep_wav_file_names.append(str(filename))

    session_css['sep_wav_file_names'] = sep_wav_file_names

    return session_css


def load_separator_model(cfg: CssCfg, is_mc: bool, models_dir: str) -> nn.Module:
    """Load multi-channel (mc) or single-channel (sc) CSS model from checkpoint and yaml files."""

    model_subpath = cfg.checkpoint_mc if is_mc else cfg.checkpoint_sc
    model_path = Path(models_dir) / model_subpath

    def fetch_one_file(path: Path, suffix: str):
        files = list(path.glob(suffix))
        if len(files) == 0:
            raise FileNotFoundError(f'expecting at least one {suffix} file in {path}')
        assert len(files) == 1, f'expecting exactly one {suffix} file in {path}'
        return str(files[0])

    yaml_path = fetch_one_file(model_path, '*.yaml')
    checkpoint_path = fetch_one_file(model_path, '*.pt')

    train_cfg = load_yaml_to_dataclass(yaml_path, TrainCfg)
    separator = get_model(train_cfg)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    def get_sub_state_dict(state_dict, prefix):
        # during training, model is wrapped in DP/DDP which introduces "module." prefix. remove it.
        return {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}

    separator.load_state_dict(get_sub_state_dict(checkpoint["model"], "module."))
    return separator


@torch.no_grad()
def separate_and_stitch(speech_mix: np.ndarray, separator: ConformerCssWrapper, fs: int,
                        device: torch.device, cfg: CssCfg) -> List[np.ndarray]:
    """
    Applies speech separation in block-online fashion.
    The long-form input is split into segments of cfg.segment_size_sec seconds with cfg.hop_size_sec hops.
    Each segment is processed by the separator and the results are stitched together.

    For multi-channel there is the option to apply MVDR, or perfrorm mask multiplication over
    the refence channel.
    For single-channel, only mask multiplication performed.

    The mask outputs of adjacent segments are aligned by considering all permutations.
    The aligned segments are underdo weighted overlap-and-add to produce the output.

    See CssCfg for various configuration options.

    Args:
        speech_mix: Input long-form speech data [Batch, Nsamples, Channels].
            Channels == 1 for single-channel.
            Channels == 7 for multi-channel (7 microphones).
        separator: Segment-wise CSS model. (see ConformerCssWrapper for example).
        fs: sample rate.
        device: torch device to use for processing each segment.
        cfg: CSS configuration.
    Returns:
        [separated_wav1, separated_wav2, ...]

    """
    assert speech_mix.ndim == 3, f'expecting 3 dimensions, got {speech_mix.shape}'

    separator.cpu()
    speech_mix = torch.from_numpy(speech_mix).cpu()

    # pass all-zeros tensor through stft to convert seconds to number of frames in stft
    dummy_in = speech_mix.new_zeros(1, int(cfg.segment_size_sec * fs), 1)
    dumm_out = separator.stft(dummy_in)  # [B, F, T, Mics]
    segment_frames = dumm_out.shape[2]
    hop_frames = int(segment_frames * cfg.hop_size_sec / cfg.segment_size_sec)
    m0_frames = int(segment_frames * cfg.seg_weight_m0_sec / cfg.segment_size_sec)
    m1_frames = int(segment_frames * cfg.seg_weight_m1_sec / cfg.segment_size_sec)
    dilation_frames = int(segment_frames * cfg.activity_dilation_sec / cfg.segment_size_sec)
    erosion_frames = int(segment_frames * cfg.activity_erosion_sec / cfg.segment_size_sec)

    # compute STFT features of the long-form mixture on cpu to avoid potential GPU memory overflow.
    stft_mix = separator.stft(speech_mix)  # [B, F, T_long, Channels], complex
    batch_size, mix_freqs, mix_frames = stft_mix.shape[:3]

    # Pad stft_mix if it is shorter than segment_frames
    if mix_frames < segment_frames:
        padding_size = segment_frames - mix_frames
        # Pad the third dimension (time frames) with zeros
        assert stft_mix.ndim == 4
        stft_mix = F.pad(stft_mix, (0, 0, 0, padding_size), mode='constant', value=0)
        mix_frames = stft_mix.shape[2]

    overlap_frames = segment_frames - hop_frames
    num_segments = int(
        np.ceil((mix_frames - overlap_frames) / hop_frames)
    )
    T = segment_frames
    pad_shape = stft_mix[:, :, :T].shape  # [B, F, T, Mics]

    separated_seg_list = []
    spk_masks_list = []

    assert not separator.training
    # segment processing happens on device
    separator.to(device)

    # I. apply separator to each segment
    range_ = trange if cfg.show_progressbar else range
    for i in range_(num_segments):
        st = i * hop_frames
        en = st + T
        if en >= mix_frames:
            # en - st < T (last segment)
            en = mix_frames
            stft_seg = stft_mix.new_zeros(pad_shape)
            t = en - st
            stft_seg[:, :, :t] = stft_mix[:, :, st:en]
        else:
            t = T
            stft_seg = stft_mix[:, :, st:en]  # [B, F, T, Channels]

        # segment-level separation
        assert batch_size == 1, 'assuming 1 example in batch. easy to support more.'

        stft_seg_device = stft_seg.to(device)
        masks: Dict[str, th.Tensor] = separator.separate(stft_seg_device)
        # dict with spk_masks, noise_masks keys

        assert masks['spk_masks'].shape[3] == cfg.num_spks
        assert stft_seg.shape[:3] == stft_seg.shape[:3]
        ref_channel = 0
        stft_seg_device_chref = stft_seg_device[:, :, :, ref_channel]
        # masks['spk_masks']:    [B, F, T, num_spks]
        # stft_seg_device:       [B, F, T, Channels]
        # stft_seg_device_chref: [B, F, T]

        num_channels = stft_seg_device.shape[3]
        if num_channels > 1 and cfg.mc_mvdr:
            mvdr_responses = make_mvdr(masks['spk_masks'].squeeze(0).moveaxis(2, 0).cpu().numpy(),
                                       masks['noise_masks'].squeeze(0).moveaxis(2, 0).cpu().numpy(),
                                       mix_stft=stft_seg_device.squeeze(0).moveaxis(2, 0).cpu().numpy(),
                                       return_stft=True)
            mvdr_responses = torch.from_numpy(np.stack(mvdr_responses, axis=-1)).unsqueeze(0).to(device)
            # [B, F, T, num_spks]
            seg_for_masking = mvdr_responses
        else:
            seg_for_masking = stft_seg_device_chref.unsqueeze(-1)  # [B, F, T, 1]

        # floored mask multiplication. if mask_floor_db == 0, mask is all-ones (assuming mask in [0, 1] range)
        mask_floor_db = cfg.mc_mask_floor_db if num_channels > 1 else cfg.sc_mask_floor_db
        assert mask_floor_db <= 0
        mask_floor = 10. ** (mask_floor_db / 20.)  # dB to amplitude
        mask_clipped = torch.clip(masks['spk_masks'], min=mask_floor)
        separated_seg = seg_for_masking * mask_clipped  # [B, F, T, num_spks]

        # Plot for debugging
        # plot_separation_methods(stft_seg_device_chref, masks, mvdr_responses, separator, cfg,
        #                         plots=['mvdr', 'masked_mvdr', 'spk_masks', 'masked_ref_ch', 'mixture'])

        if cfg.normalize_segment_power:
            # normalize to match the input mixture power
            assert stft_seg_device_chref.ndim == 3
            mix_energy = torch.sqrt(
                torch.mean(stft_seg_device_chref[:, :, :t].abs().pow(2),  # squared mag
                           dim=(1, 2), keepdim=True)
            )

            assert torch.is_complex(separated_seg)
            sep_energy = torch.sqrt(
                torch.mean(separated_seg[:, :, :t].sum(-1).abs().pow(2),  # sum over spks, squared mag
                           dim=(1, 2), keepdim=True
                )
            )
            separated_seg = (mix_energy / sep_energy)[..., None]  * separated_seg

        separated_seg_list.append(separated_seg.cpu())         # [B, F, T, num_spks]
        spk_masks_list.append(masks['spk_masks'].cpu())  # [B, F, T, num_spks]


    # stitch the separated segments together
    stft_stitched = stft_mix.new_zeros(*stft_mix.shape[:3], cfg.num_spks)   # [B, F, T_long, num_spks]
    mask_stitched = stft_mix.new_zeros(*stft_mix.shape[:3], cfg.num_spks, dtype=torch.float)
    wg_stitched = stft_mix.new_zeros(mix_frames, dtype=torch.float32)  # [T_long]
    # add first segment
    wg_seg = calc_segment_weight(segment_frames, m0_frames, m1_frames, is_first_seg=True)
    wg_stitched[:segment_frames] += wg_seg
    stft_stitched[:, :, :segment_frames] += wg_seg.view(1, 1, -1, 1) * separated_seg_list[0]
    mask_stitched[:, :, :segment_frames] += wg_seg.view(1, 1, -1, 1) * spk_masks_list[0]

    pit = PitWrapper({'mse': mse_loss, 'l1': l1_loss}[cfg.stitching_loss])

    # II. stitch the separated segments together
    for i in range(1, num_segments):
        if cfg.stitching_input == 'mask':
            left_input, right_input = spk_masks_list[i-1], spk_masks_list[i]
        elif cfg.stitching_input == 'separation_result':
            # masked magnitudes
            left_input, right_input = separated_seg_list[i - 1].abs(), separated_seg_list[i].abs()
        else:
            assert False, f'unexpected stitching_input: {cfg.stitching_input}'

        assert left_input.shape[2] == right_input.shape[2] == segment_frames
        loss, right_perm = pit(left_input[:, :, -overlap_frames:], right_input[:, :, :overlap_frames])

        # Plot for debugging:
        # plot_left_right_stitch(separator, left_input, right_input, right_perm,
        #                        overlap_frames, cfg, stft_seg_to_play=separated_seg_list[i][..., 0], fs=fs)

        # permute current segment to match with the previous one
        for ib in range(batch_size):
            spk_masks_list[i][ib] = spk_masks_list[i][ib, ..., right_perm[ib]]
            separated_seg_list[i][ib] = separated_seg_list[i][ib, ..., right_perm[ib]]

        st = i * hop_frames
        en = min(st + segment_frames, mix_frames)
        # weighted overlap-and-add
        wg_seg = calc_segment_weight(segment_frames, m0_frames, m1_frames, is_last_seg=(i==num_segments-1))
        wg_seg = wg_seg[:en-st]  # last segment may be shorter
        wg_stitched[st:en] += wg_seg
        assert torch.is_complex(separated_seg_list[i]), 'summation assumes complex representation'
        stft_stitched[:, :, st:en] += wg_seg.view(1, 1, -1, 1) * separated_seg_list[i][:, :, :en-st]
        mask_stitched[:, :, st:en] += wg_seg.view(1, 1, -1, 1) * spk_masks_list[i][:, :, :en-st]

    assert (wg_stitched > 1e-5).all(), 'zero weights found. check hop_size, segment_size or m0, m1'
    stft_stitched /= wg_stitched.view(1, 1, -1, 1)
    mask_stitched /= wg_stitched.view(1, 1, -1, 1)

    # III. apply temporal segmentation mask
    assert batch_size == 1
    activity = mask_stitched.mean(dim=1)[0]  # [T, num_spks]
    activity_b = activity >= cfg.activity_th
    # dilate -> erode each speaker activity
    activity_final = [torch.from_numpy(erode(dilate(x.numpy(), dilation_frames), erosion_frames))
                      for x in th.unbind(activity_b, dim=1)]
    activity_final = th.stack(activity_final, dim=1)[None]  # [B, T, num_spks]

    assert activity_final.shape[1:] == stft_stitched.shape[2:]
    # apply segmentation mask to stft_stitched
    stft_stitched = activity_final.unsqueeze(1) * stft_stitched
    # potential optimization: drop silent parts to save ASR compute. requires keeping original time.

    # [B, F, T, num_spks] -> [B*num_spks, F, T]
    stft_stitched = (stft_stitched.moveaxis(3, 1).contiguous()
                     .view(batch_size*cfg.num_spks, mix_freqs, mix_frames ))
    separator.cpu()
    separated_wavs = separator.istft(stft_stitched).view(batch_size, cfg.num_spks, -1).numpy()
    # [B, num_spks, Nsamples]
    separated_wavs = np.split(separated_wavs, cfg.num_spks, axis=1)
    assert batch_size == 1
    separated_wavs = [w.squeeze() for w in separated_wavs]  # num_spks list of [Nsamples]
    assert len(separated_wavs) == cfg.num_spks

    # Plot for debugging:
    # plot_stitched_masks(mask_stitched, activity_b, activity_final, cfg)
    # play_wav(separated_wavs[2], fs, volume_factor=5.)
    # play_wav(speech_mix[0,:, 0].cpu().numpy(), fs, volume_factor=5.)

    return separated_wavs


def calc_segment_weight(seg_frames: int, m0_frames: int, m1_frames: int,
                        is_first_seg: bool = False, is_last_seg: bool = False):

    """
    Returns weighting for segment.

    During weighted overlap-and-add the separated segments will be weighted by a time-window defined
    by m0 and m1 parameters.
    Frames 0 to m0_frames will have weight=0, m1_frames and onward will have weight=1.
    Frames between m0_frames and m1_frames will have linearly increasing weight.
    The weights on the right side will behave symetrically.

        Weight
    1     |            ____________
          |           /            \
          |          /              \
          |         /                \
    0     |________/                  \________
          0      m0  m1           m1' m0'
                 <---->           <---->
             Linear Increase   Linear Decrease

    m1' = seg_frames - m1
    m0' = seg_frames - m0


    Args:
        seg_frames: segment length in frames
        m0_frames: start of linear increase
        m1_frames: end of linear increase
        is_first_seg: True if this is the first segment in the long-form audio
        is_last_seg: True if this is the last segment in the long-form audio
    """
    assert seg_frames > 2 * m1_frames, \
        'not enough frames to fit weighting window. try modifying hop_size, segment_size or m0, m1'
    wg_win = torch.ones(seg_frames, dtype=torch.float32)
    wg_win[:m0_frames] = 0
    wg_win[len(wg_win)-m0_frames:] = 0
    linear = torch.linspace(0.1, 1, m1_frames - m0_frames)  # linear transition from 0.1 to 1
    wg_win[m0_frames:m1_frames] = linear
    wg_win[-m1_frames:-m0_frames] = torch.flip(linear, (0,))

    if is_first_seg:
        # first segment is the only contributor to its left edge, so we can't have zero weight.
        wg_win[:m0_frames] = 0.1
    if is_last_seg:
        # similar to the above, last segment is the only contirbutor to its right edge.
        wg_win[len(wg_win) - m0_frames:] = 0.1

    return wg_win
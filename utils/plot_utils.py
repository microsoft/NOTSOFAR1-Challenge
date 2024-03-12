"""Plot CSS inference intermediate results for debug. See usage in css.py"""
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from utils.audio_utils import play_wav, write_wav


def plot_stitched_masks(mask_stitched, activity_b, activity_final, cfg):
    import matplotlib.pyplot as plt
    activity = mask_stitched.mean(dim=1)  # [B, T, num_spks]
    total_plots = cfg.num_spks * 2
    time_frames = mask_stitched.size(2)  # Assuming the number of time frames is the third dimension
    plt.figure(figsize=(15, 5 * total_plots))
    for j in range(cfg.num_spks):
        # Plot for mask_stitched
        plt.subplot(total_plots, 1, 2 * j + 1)
        plt.imshow(mask_stitched[0, :, :, j], aspect='auto', origin='lower')
        # plt.colorbar()
        plt.title(f"Speaker {j + 1} Mask")
        # plt.xlabel("Time Frames")
        plt.ylabel("Frequency Bins")
        plt.xlim(0, time_frames - 1)  # Set x-axis limits

        # Plot for activity
        plt.subplot(total_plots, 1, 2 * j + 2)
        plt.plot(activity[0, :, j], label='mean mask')
        plt.plot(activity_b[:, j], label=f'thresh={cfg.activity_th}')
        plt.plot(activity_final[0, :, j],
                 label=f'dilate({cfg.activity_dilation_sec})->erode({cfg.activity_erosion_sec})')
        plt.title(f"Speaker {j + 1} Activity")
        # plt.xlabel("Time Frames")
        plt.ylabel("Average Activity")
        plt.xlim(0, time_frames - 1)  # Set x-axis limits to be the same as the mask_stitched plot
        plt.ylim(0, 1.05)
        plt.legend(loc='best')  # Add a legend
    plt.suptitle('Speaker Masks and Activities')
    plt.show()


def plot_left_right_stitch(separator, left_input, right_input, right_perm, overlap_frames,
                           cfg, stft_seg_to_play: Optional[torch.Tensor]=None, fs: Optional[int]=None):
    if stft_seg_to_play is not None:
        separator.cpu()
        wav = separator.istft(stft_seg_to_play).cpu().numpy()
        play_wav(wav.squeeze(), fs, volume_factor=5.)

    left = left_input    # overlapping part - [:, :, -overlap_frames:]
    right = right_input  # overlapping part - [:, :, :overlap_frames]
    import matplotlib.pyplot as plt
    num_spks = cfg.num_spks
    plt.figure(figsize=(15, 5 * num_spks))
    for j in range(num_spks):
        plt.subplot(num_spks, 1, j + 1)
        plt.imshow(left[0, :, :, j], aspect='auto', origin='lower')
        plt.axvline(x=left.shape[2] - overlap_frames, color='red', linestyle='--')
        plt.colorbar()
        plt.title(f"Speaker {j + 1} Mask")
        plt.xlabel("Time Frames")
        plt.ylabel("Frequency Bins")
    plt.suptitle('left')
    plt.show()
    plt.figure(figsize=(15, 5 * num_spks))
    for j in range(num_spks):
        plt.subplot(num_spks, 1, j + 1)
        plt.imshow(right[0, :, :, right_perm[0][j]], aspect='auto', origin='lower')
        plt.axvline(x=overlap_frames, color='red', linestyle='--')
        plt.colorbar()
        plt.title(f"Speaker {j + 1} Mask")
        plt.xlabel("Time Frames")
        plt.ylabel("Frequency Bins")
    plt.suptitle('right')
    plt.show()


def plot_separation_methods(stft_seg_device_chref, masks, mvdr_responses, separator, cfg, plots):
    """Plot various masking methods for multi-channel segment, and writes them as wav files.
    plots arg controls what to plot.

    For full plot:
        plots = ['mvdr', 'masked_mvdr', 'spk_masks', 'masked_ref_ch', 'mixture']
    """
    import matplotlib.pyplot as plt
    import librosa
    plots_ordered = []
    num_spks = cfg.num_spks
    fig, axs = plt.subplots(num_spks, len(plots), figsize=(30, 5 * num_spks))
    masked_ref_ch = stft_seg_device_chref.unsqueeze(-1) * masks['spk_masks']
    masked_mvdr = mvdr_responses * masks['spk_masks']  # note, no floor
    col_ind = -1
    if 'mvdr' in plots:
        plots_ordered.append('mvdr')
        col_ind += 1
        for j in range(num_spks):
            ax = axs[j, col_ind]
            img = librosa.display.specshow(
                librosa.amplitude_to_db(mvdr_responses[0, :, :, j].abs().cpu(), ref=np.max),
                y_axis='linear', x_axis='time', ax=ax, sr=16000)
            ax.set_title(f'Speaker {j + 1} Spectrogram')
            plt.colorbar(img, ax=ax, format="%+2.0f dB")
            ax.set_xlabel("Time Frames")
            ax.set_ylabel("Frequency Bins")
    if 'masked_mvdr' in plots:
        plots_ordered.append('masked_mvdr')
        col_ind += 1
        for j in range(num_spks):
            ax = axs[j, col_ind]
            img = librosa.display.specshow(
                librosa.amplitude_to_db(masked_mvdr[0, :, :, j].abs().cpu(), ref=np.max),
                y_axis='linear', x_axis='time', ax=ax, sr=16000)
            ax.set_title(f'Speaker {j + 1} Spectrogram')
            plt.colorbar(img, ax=ax, format="%+2.0f dB")
            ax.set_xlabel("Time Frames")
            ax.set_ylabel("Frequency Bins")
    if 'masked_ref_ch' in plots:
        plots_ordered.append('masked_ref_ch')
        col_ind += 1
        for j in range(num_spks):
            ax = axs[j, col_ind]
            img = librosa.display.specshow(
                librosa.amplitude_to_db(masked_ref_ch[0, :, :, j].abs().cpu(), ref=np.max),
                y_axis='linear', x_axis='time', ax=ax, sr=16000)
            ax.set_title(f'Speaker {j + 1} Spectrogram')
            plt.colorbar(img, ax=ax, format="%+2.0f dB")
            ax.set_xlabel("Time Frames")
            ax.set_ylabel("Frequency Bins")
    if 'spk_masks' in plots:
        plots_ordered.append('spk_masks')
        col_ind += 1
        for j in range(num_spks):
            ax = axs[j, col_ind]
            img = ax.imshow(masks['spk_masks'][0, :, :, j].cpu(), aspect='auto', origin='lower', vmin=0,
                            vmax=1)
            plt.colorbar(img, ax=ax)
            ax.set_xlabel("Time Frames")
            ax.set_ylabel("Frequency Bins")
    if 'mixture' in plots:
        plots_ordered.append('mixture')
        col_ind += 1
        # plot mixture ch0
        ax = axs[0, col_ind]
        img_right = librosa.display.specshow(
            librosa.amplitude_to_db(stft_seg_device_chref[0, :, :].abs().cpu(), ref=np.max),
            y_axis='linear', x_axis='time', ax=ax)
        plt.colorbar(img_right, ax=ax, format="%+2.0f dB")
        ax.set_xlabel("Time Frames")
        ax.set_ylabel("Frequency Bins")

        # plot noisemask
        ax = axs[1, col_ind]
        img = ax.imshow(masks['noise_masks'][0, :, :, 0].cpu(), aspect='auto', origin='lower', vmin=0, vmax=1)
        plt.colorbar(img, ax=ax)
        ax.set_xlabel("Time Frames")
        ax.set_ylabel("Frequency Bins")

    plt.suptitle(' | '.join(plots_ordered))
    plt.tight_layout()
    plt.show()

    istft = lambda x: separator.istft(x).cpu().numpy()[0]
    # x: [B, num_spks, Nsamples]
    out_dir = Path('artifacts/analysis/separated_seg')
    write_wav(out_dir / 'input_ref_ch.wav', samps=istft(stft_seg_device_chref), sr=16000)
    for j in range(num_spks):
        write_wav(out_dir / f'masked_ref_ch{j}.wav', samps=istft(masked_ref_ch[..., j]), sr=16000)
        write_wav(out_dir / f'mvdr_{j}.wav', samps=istft(mvdr_responses[..., j]), sr=16000)
        write_wav(out_dir / f'masked_mvdr_{j}.wav', samps=istft(masked_mvdr[..., j]), sr=16000)
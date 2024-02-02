"""Plot CSS inference intermediate results for debug. See usage in css.py"""

from typing import Optional

import torch

from utils.audio_utils import play_wav


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
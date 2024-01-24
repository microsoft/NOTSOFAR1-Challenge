import os
import shutil
import pandas as pd
import numpy as np

from utils.audio_utils import read_wav, write_wav
from diarization.diarization_common import DiarizationCfg
from diarization.time_based_diarization import time_based_diarization
from diarization.word_based_diarization import word_based_clustering


def diarization_inference(out_dir: str, segments_df: pd.DataFrame, cfg: DiarizationCfg,
                          fetch_from_cache: bool, simulate_css: bool=False) -> pd.DataFrame:
    """
    Run diarization to assign a speaker label to each ASR word.

    Two diarization modes are supported:
    1. Pre-SR diarization that runs diarization without the knowledge of ASR.
        In this mode, we directly call NeMo's diarization recipes, such as NMESC or NMESCC
        followed by MSDD. Then, for each ASR word, the speaker that is the most active within
        the word's time boundaries is assigned to the word.
        Set cfg.method to "nmesc" to use NMESC recipe of NeMo in the config file.
        Set cfg.method to "nmesc_msdd" to use the NMESC followed by MSDD recipe of NeMo.
    2. Post-SR diarization that runs diarization after ASR. Allows the use of word boundaries.
        In this mode, we extract a speaker embedding vector for each word, and then call
        NeMo's NMESC for clustering. We also adopted the multi-scale speaker embedding window
        concept from NeMo, and extract multiple scale speaker embedding vectors for each word,
        each scale using different window sizes. The final affinity matrix is a simple average
        of the affinity matrixces of all the scales.
        To use this mode, set cfg.method to "word_nmesc".

    Args:
        out_dir: the directory to store generated files in the diarization step.
            This allows the cache of files and skip some steps when the code is run again.
        segments_df: a dataframe of transcribed segments for a given session, with columns:
            'start_time': start time of the segment in seconds.
            'end_time': end time of the segment in seconds.
            'text': the text of the segment.
            'word_timing': a list of [word, start, end] lists.
            'meeting_id': the meeting id.
            'session_id': the session id.
        cfg: diarization configuration.
        overwrite: whether to overwrite previously generated diarizaiton files
        simulate_css: a temporary option that simulates multiple CSS unmixed channels
            so we can test whether diarization works with CSS outputs.
    Returns:
        attributed_segments_df: a new set of segments with 'speaker_id' column added.
    """

    assert segments_df.session_id.nunique() <= 1, 'no cross-session information is permitted'
    assert segments_df.wav_file_names.nunique() <= 3, 'expecting at most three separated channels'

    session_name = segments_df.session_id[0]
    
    output_dir = os.path.join(out_dir, "diarization", session_name, cfg.method)
    assert not fetch_from_cache, 'needs support'
    os.makedirs(output_dir, exist_ok=True)
    
    if simulate_css:
        # simulate CSS audio channels
        # randomly assign the ASR segments into one of the three CSS unmixed channels
        seg = segments_df.iloc[0]
        sr, wav = read_wav(seg['wav_file_names'], normalize=True, return_rate=True)
        channel_segments = [[], [], []]
        new_seg_wav_files = []
        for _, seg in segments_df.iterrows():
            channel_id = np.random.randint(0, 3)
            channel_segments[channel_id].append([seg.start_time, seg.end_time])
            new_seg_wav_files.append(os.path.join(output_dir, f"ch{channel_id}.wav"))
        segments_df.wav_file_names = new_seg_wav_files
        
        css_wavs = np.zeros((3, wav.size))
        for channel_id, seg_times in enumerate(channel_segments):
            for seg in seg_times:
                idx1 = int(seg[0]*sr)
                idx2 = int(seg[1]*sr)
                css_wavs[channel_id][idx1:idx2] = wav[idx1:idx2]
        for channel_id in range(3):
            write_wav(os.path.join(output_dir, f"ch{channel_id}.wav"), css_wavs[channel_id])   
        # end of simulation
    
    wav_files_sorted = sorted(segments_df.wav_file_names.unique())
    
    if cfg.method == "word_nmesc":
        attributed_segments_df = word_based_clustering(wav_files_sorted, segments_df, cfg)
    else:
        attributed_segments_df = time_based_diarization(wav_files_sorted, segments_df, output_dir, cfg)
    
    return attributed_segments_df

import os
from typing import Optional
import pandas as pd
from pathlib import Path

from diarization.diarization_common import DiarizationCfg
from diarization.time_based_diarization import time_based_diarization
from diarization.word_based_diarization import word_based_clustering
from utils.logging_def import get_logger
from utils.torch_utils import get_world_size

_LOG = get_logger('diarization')


def diarization_inference(out_dir: str, segments_df: pd.DataFrame, cfg: DiarizationCfg,
                          fetch_from_cache: bool, device: Optional[str] = None) -> pd.DataFrame:
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

    A known limitation of the diarization baseline is that the words from the CSS streams
    are pooled and clustered, and stream ID is not used in clustering. It is possible that
    words from different streams that overlap in time are assigned to the same speaker.
    This will trigger warning in tcp_wer and tcorc_wer computation and potentially degrade results.

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
            'wav_file_name': the name of the wav file that the segment was transcribed from.
                this is typically points to the speech separated wav file (see CSS module).
        cfg: diarization configuration.
        fetch_from_cache: If True, returns the cached results if they exist. Otherwise, runs the inference.
        device: the device to use for loading the model and running inference.
    Returns:
        attributed_segments_df: a new set of segments with 'speaker_id' column added.
    """

    _LOG.info("Running Speaker Diarization")

    assert segments_df.session_id.nunique() <= 1, 'no cross-session information is permitted'

    # these two modes are for debugging and analysis
    if cfg.method == "skip":
        _LOG.info("Skipping Diarization")
        attributed_segments_df = segments_df.copy()
        attributed_segments_df['speaker_id'] = 'spk0'
        return attributed_segments_df
    elif cfg.method == "by_wav_file_name":
        attributed_segments_df = segments_df.copy()
        # map each unique wav_file_name to an index
        wav_file_name_ind, uniques = pd.factorize(attributed_segments_df['wav_file_name'], sort=True)
        attributed_segments_df['speaker_id'] = wav_file_name_ind
        attributed_segments_df['speaker_id'] = 'wav_' + attributed_segments_df['speaker_id'].astype(str)
        _LOG.info(f"Diarization by wav file names: {uniques}")
        return attributed_segments_df

    session_name = segments_df.session_id[0]
    is_ct = session_name.startswith('close_talk')
    assert segments_df.wav_file_name.nunique() <= 3 or is_ct, 'expecting at most three separated channels'
    output_dir = Path(out_dir) / "diarization" / session_name / cfg.method
    out_file = output_dir / "all_segments_df.pkl"

    # Skip cache and writing ops if running in DDP mode, it is necessary to continue evaluate the model on each device
    skip_cache_and_write = get_world_size() > 1

    if not skip_cache_and_write:
        if fetch_from_cache and out_file.exists():
            attributed_segments_df = pd.read_pickle(out_file)
            return attributed_segments_df
        os.makedirs(output_dir, exist_ok=True)

    segments_df = segments_df.copy()
    # wav_file_name as category to convert to indices
    segments_df['wav_file_name'] = segments_df['wav_file_name'].astype('category')
    assert 'wav_file_name_ind' not in segments_df
    segments_df['wav_file_name_ind'] = segments_df['wav_file_name'].cat.codes
    wav_files = segments_df['wav_file_name'].cat.categories.to_list()
    
    if cfg.method == "word_nmesc":
        attributed_segments_df = word_based_clustering(wav_files, segments_df, cfg, device)
    else:
        attributed_segments_df = time_based_diarization(wav_files, segments_df, str(output_dir), cfg)
    
    if not skip_cache_and_write:
        attributed_segments_df.to_pickle(out_file)
        _LOG.info(f'Speaker Diarization saved to {out_file}')

    return attributed_segments_df

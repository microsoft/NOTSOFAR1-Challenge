import os
import pandas as pd
from pathlib import Path

from diarization.diarization_common import DiarizationCfg
from diarization.time_based_diarization import time_based_diarization
from diarization.word_based_diarization import word_based_clustering
from utils.logging_def import get_logger

_LOG = get_logger('diarization')


def diarization_inference(out_dir: str, segments_df: pd.DataFrame, cfg: DiarizationCfg,
                          fetch_from_cache: bool) -> pd.DataFrame:
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
            'wav_file_name': the name of the wav file that the segment was transcribed from.
        cfg: diarization configuration.
        fetch_from_cache: If True, returns the cached results if they exist. Otherwise, runs the inference.
    Returns:
        attributed_segments_df: a new set of segments with 'speaker_id' column added.
    """

    _LOG.info("Running Speaker Diarization")

    assert segments_df.session_id.nunique() <= 1, 'no cross-session information is permitted'
    assert segments_df.wav_file_name.nunique() <= 3, 'expecting at most three separated channels'

    session_name = segments_df.session_id[0]
    
    output_dir = Path(out_dir) / "diarization" / session_name / cfg.method
    out_file = output_dir / "all_segments_df.pkl"

    if fetch_from_cache and out_file.exists():
        attributed_segments_df = pd.read_pickle(out_file)
        return attributed_segments_df

    wav_files_sorted = sorted(segments_df.wav_file_name.unique())
    os.makedirs(output_dir, exist_ok=True)

    if cfg.method == "word_nmesc":
        attributed_segments_df = word_based_clustering(wav_files_sorted, segments_df, cfg)
    else:
        attributed_segments_df = time_based_diarization(wav_files_sorted, segments_df, str(output_dir), cfg)

    attributed_segments_df.to_pickle(out_file)
    _LOG.info(f'Speaker Diarization saved to {out_file}')

    return attributed_segments_df

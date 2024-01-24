import glob
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import whisper
from whisper import Whisper
from scipy.io import wavfile
from whisper.normalizers import EnglishTextNormalizer


@dataclass
class WhisperAsrCfg:
    # TODO try large-v3
    model_name: str = 'large-v2' # use 'large-v2' for experiments, use 'tiny' for fast debugging
    language: Optional[str] = 'en'  # language that the speech is in (if None, whisper runs language ID)
    word_level_time_stamps: bool = True

    def text_normalizer(self):
        return EnglishTextNormalizer()

    def assert_valid(self):
        assert self.model_name in ['tiny.en', 'tiny', 'base.en', 'base', 'small.en', 'small', 'medium.en',
                                'medium', 'large-v1', 'large-v2', 'large-v3', 'large']


def asr_inference(out_dir: str, session: pd.Series, cfg: WhisperAsrCfg, fetch_from_cache: bool):
    """
    Applies automatic speech recognition using Whisper - an ASR model by OpenAI.

    Args:
        out_dir: the outputs per module are saved to out_dir/{module_name}/{session_id}.
        session: Row representing session to evaluate.
        cfg: Specifies Whisper's configuration (paths, model parameters, etc).
        fetch_from_cache: If True, returns the cached results if they exist. Otherwise, runs the inference.
    Returns:
        segments_df: a dataframe of transcribed segments returned by ASR with the following columns:
            'start_time': start time of the segment in seconds.
            'end_time': end time of the segment in seconds.
            'text': the text of the segment.
            'word_timing': a list of [word, start, end] lists.
            'meeting_id': the meeting id.
            'session_id': the session id.
    """
    cfg.assert_valid()
    decode_options = dict(language=cfg.language, word_timestamps=cfg.word_level_time_stamps)
    transcribe_options = dict(task="transcribe", **decode_options)

    # TODO use wav streams from CSS output. Keep both local and global timestamps?
    wav_files = session.sep_wav_file_names
    assert isinstance(wav_files, list) and len(wav_files) == 1, \
        'currently only 1 stream supported. make sure to set css_inference to pass_through_mic0=True'
    wav_file = wav_files[0]
    assert isinstance(wav_file, str)

    json_file = os.path.join(out_dir, "asr", session.session_id, cfg.model_name, "reco.json")
    if fetch_from_cache and os.path.isfile(json_file):
        with open(json_file, "r") as file:
            results = json.load(file)
    else:
        model = whisper.load_model(cfg.model_name)
        results =  model.transcribe(wav_file, **transcribe_options)
        os.makedirs(os.path.dirname(json_file), exist_ok=True)
        with open(json_file, "w") as file:
            json.dump(results, file, indent=4)
    raw_seg_df = pd.DataFrame(results['segments'])

    # each entry in this column is a list of [word, start, end] lists
    word_start_end = raw_seg_df['words'].apply(lambda x: [[w['word'], w['start'], w['end']] for w in x])

    segments_df = pd.DataFrame(
        {'start_time': raw_seg_df['start'],
         'end_time': raw_seg_df['end'],
         'text': raw_seg_df['text'],
         'word_timing': word_start_end})

    segments_df['meeting_id'] = session.meeting_id
    segments_df['session_id'] = session.session_id
    segments_df['wav_file_names'] = wav_file  # TODO take from CSS output

    return segments_df


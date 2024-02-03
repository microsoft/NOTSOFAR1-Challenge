from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
import whisper
from tqdm import tqdm

from utils.logging_def import get_logger
from utils.text_norm_whisper_like import get_txt_norm

_LOG = get_logger('asr')


@dataclass
class WhisperAsrCfg:
    model_name: str = 'large-v2'  # use 'large-v2' for experiments, use 'tiny' for fast debugging
    language: Optional[str] = 'en'  # language that the speech is in (if None, whisper runs language ID)
    word_level_time_stamps: bool = True
    beam_size: Optional[int] = 5
    hallucination_silence_threshold: Optional[float] = 2.

    def text_normalizer(self):
        return get_txt_norm("chime8")

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
            'wav_file_name': the name of the wav file that the segment was transcribed from.
    """
    _LOG.info('Running ASR')
    cfg.assert_valid()
    decode_options = dict(language=cfg.language,
                          word_timestamps=cfg.word_level_time_stamps,
                          beam_size=cfg.beam_size,
                          hallucination_silence_threshold=cfg.hallucination_silence_threshold)
    transcribe_options = dict(task="transcribe", **decode_options)

    wav_files = session.sep_wav_file_names
    assert isinstance(wav_files, list)

    out_file = Path(out_dir) / 'asr' / session.session_id / cfg.model_name / "all_segments_df.pkl"

    if fetch_from_cache and out_file.exists():
        _LOG.info(f'Loading ASR results from {out_file}')
        all_segments_df = pd.read_pickle(out_file)
        return all_segments_df

    _LOG.info(f'Loading Whisper model: {cfg.model_name}')
    model = whisper.load_model(cfg.model_name)

    _LOG.info(f'Running ASR on {len(wav_files)} streams')
    segments_dfs = []
    for wav_file in tqdm(wav_files, desc='running ASR'):
        results = model.transcribe(str(wav_file), **transcribe_options)
        if len(results['segments']) == 0:
            _LOG.warning(f'No segments returned for {wav_file}')
            continue
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
        segments_df['wav_file_name'] = wav_file

        segments_dfs.append(segments_df)

    all_segments_df = pd.concat(segments_dfs, ignore_index=True)

    out_file.parent.mkdir(parents=True, exist_ok=True)
    all_segments_df.to_pickle(out_file)
    _LOG.info(f'ASR results saved to {out_file}')

    return all_segments_df


import json
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import soundfile
from tqdm import tqdm

from utils.audio_utils import write_wav
from utils.torch_utils import is_zero_rank, barrier


def load_data(meetings_dir: str, session_query: Optional[str] = None,
              return_close_talk: bool = False, out_dir: Optional[str] = None
              ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load all meetings from the meetings dir

    Args:
        meetings_dir: directory containing meetings.
            Example: project_root/artifacts/meeting_data/dev_set/240121_dev/MTG/
        session_query: a query string to filter the sessions (optional)
            When submitting results, this should be None so no filtering occurs.
        return_close_talk: if True, return each meeting as a session with all close-talk devices as its
            wav_file_names.
            Close-talk must not be used during inference. However, this can be used as supervision
            signal during training or for analysis.
        out_dir: directory to save outputs to. only used when return_close_talk is True.
    Returns:
        all_session_df (per device):
            Each line corresponds to a recording of a meeting captured with a single device
            (referred to as a 'session').
            If a meeting was recorded with N devices (single or multi-channel), the DataFrame should contain
            N lines – one for every device recording.
            Rules:
            - Inference must run independently for each session (device) and no cross-session information
                is permitted.
            - Use of close-talk microphones is not permitted during inference.
        all_gt_utt_df (per utt):
            each line is a ground truth utterance
        all_gt_metadata_df (per meeting):
            each line is a meeting's metadata: participants, topics,
            hashtags (#WalkAndTalk, #TalkNearWhiteboard etc. useful for analysis) and more.
    """
    meetings_dir = Path(meetings_dir)

    # list to store dataframes for each meeting
    gt_utt_dfs = []
    session_dfs = []
    metadata_dfs = []

    sorted_dirs = sorted(meetings_dir.glob('*/'))
    for meeting_subdir in tqdm(sorted_dirs, desc='loading meetings data'):
        if not meeting_subdir.is_dir():
            continue
        transcription_file = meeting_subdir / 'gt_transcription.json'
        devices_file = meeting_subdir / 'devices.json'
        metadata_file = meeting_subdir / 'gt_meeting_metadata.json'

        gt_utt_df = None
        if transcription_file.exists():
            # we have GT transcription
            gt_utt_df = pd.read_json(transcription_file)
            # add a 'meeting_id' column
            gt_utt_df['meeting_id'] = meeting_subdir.name
            gt_utt_dfs.append(gt_utt_df)

        if metadata_file.exists():
            with open(metadata_file, 'r') as file:
                metadata = json.load(file)
            metadata_df = pd.DataFrame([metadata])
            metadata_dfs.append(metadata_df)

        devices_df = pd.read_json(devices_file)
        devices_df['meeting_id'] = meeting_subdir.name
        if return_close_talk:
            devices_df = devices_df[devices_df.is_close_talk].copy()
            assert len(devices_df) > 0, 'no close-talk devices found'
            assert gt_utt_df is not None, 'expecting GT transcription'

            new_wav_file_names = concat_speech_segments(devices_df, gt_utt_df, meeting_subdir, out_dir)

            # original close-talk:
            # orig_wav_file_names = devices_df.wav_file_names.apply(lambda x: str(meeting_subdir / x)).to_list()

            devices_df = devices_df.iloc[0:1].copy()
            devices_df['device_name'] = 'close_talk'
            devices_df['wav_file_names'] = [new_wav_file_names]  # orig_wav_file_names
            devices_df['session_id'] = 'close_talk/' + meeting_subdir.name
        else:
            # drop close-talk devices
            devices_df = devices_df[~devices_df.is_close_talk].copy()

            prefix = devices_df.is_mc.map({True: 'multichannel', False: 'singlechannel'})
            devices_df['session_id'] = prefix + '/' + meeting_subdir.name + '_' + devices_df['device_name']
            # convert to a list of full paths by appending meeting_subdir to each file in wav_file_name
            devices_df['wav_file_names'] = devices_df['wav_file_names'].apply(
                lambda x: [str(meeting_subdir / file_name.strip()) for file_name in x.split(',')]
            )

        session_dfs.append(devices_df)


    # concatenate all meetings into one big DataFrame
    all_gt_utt_df = pd.concat(gt_utt_dfs, ignore_index=True) if gt_utt_dfs else None
    all_session_df = pd.concat(session_dfs, ignore_index=True)
    all_metadata_df = pd.concat(metadata_dfs, ignore_index=True) if metadata_dfs else None

    # MtgType column is useful for querying, but it is on the metadata df. merge it into session df.
    if all_metadata_df is not None:
        merged_df = all_session_df.merge(all_metadata_df[['meeting_id', 'MtgType']],
                                         on='meeting_id', how='inner')
        assert len(merged_df) == len(all_session_df)
        assert not merged_df.MtgType.isna().any(), 'expecting valid MtgType values'
        all_session_df = merged_df
        assert not all_session_df.MtgType.str.startswith("read").any(), \
            '"read" meetings are for debug, they are not expected here'
        # avoid using MtgType from here on
        all_session_df.drop('MtgType', axis=1, inplace=True)

    if session_query:
        query, process_first_n = _process_query(session_query)
        all_session_df.query(query, inplace=True)
        if process_first_n:
            all_session_df = all_session_df.head(process_first_n)

    return all_session_df, all_gt_utt_df, all_metadata_df


def _process_query(query):
    """ Split query into a few parts
        Query can have the following format:
        1. "query_string"
        2. "query_string ##and index<n##"
           After executing "query_string" the index is not relevant anymore, it can be affected by the executed query,
           and some of the rows of the original df can be removed. Hence if we want to get only the first n rows,
           we must use head(n) after executing the first query part.
    """
    if query.endswith('##'):
        first_query = query.split('##')[0]
        process_first_n = query.split('##')[1].split('<')[-1]
        return first_query, int(process_first_n)
    return query, None


def concat_speech_segments(devices_df, gt_utt_df, meeting_subdir: Path, out_dir: str,
                           silence_duration_sec: float = 0.):
    """
    Concatenates segmented speech segments from close-talk audio files specified in `devices_df`,
    inserting a specified duration of silence between segments (silence_duration_sec), and adjusts the
    timing information in `gt_utt_df` accordingly.
    """
    meeting_id = devices_df.meeting_id.unique().item()
    assert gt_utt_df.meeting_id.unique().item() == meeting_id

    # Process each wav to concatenate all speech segments and silence, and adjust timings in gt_utt_df
    new_wav_file_names = []
    for wav_file_name in devices_df['wav_file_names']:
        gt_utt_df_cur = gt_utt_df[gt_utt_df['ct_wav_file_name'] == wav_file_name]
        assert gt_utt_df_cur.start_time.is_monotonic_increasing

        # Track cumulative samples to adjust start and end times
        cumulative_secs = 0.
        new_wav_segments = []
        wav, sr = soundfile.read(meeting_subdir / wav_file_name, dtype='float32')

        # Silence duration between segments
        silence_duration_samples = int(silence_duration_sec * sr)
        silence = np.zeros(silence_duration_samples, dtype=wav.dtype)

        for index, row in gt_utt_df_cur.iterrows():
            segment = wav[int(row.start_time * sr):int(row.end_time * sr)]
            # Append the current speech segment and silence
            new_wav_segments.append(segment)
            new_wav_segments.append(silence)

            # Update timings in gt_utt_df
            delta_t = cumulative_secs - row.start_time
            gt_utt_df.at[index, 'start_time'] += delta_t
            gt_utt_df.at[index, 'end_time'] += delta_t
            gt_utt_df.at[index, 'word_timing'] = [[w, s + delta_t, e + delta_t]
                                                  for w, s, e in row.word_timing]

            cumulative_secs += row.end_time - row.start_time + silence_duration_sec

        # Concatenate all speech and silence segments
        new_wav = np.concatenate(new_wav_segments)

        new_file_name = str(Path(out_dir) / 'concat_close_talk'/ meeting_id / f'{wav_file_name}')
        new_wav_file_names.append(new_file_name)
        if is_zero_rank():
            print(f'{new_file_name=}')
            write_wav(new_file_name, samps=new_wav, sr=sr)

    barrier()
    return new_wav_file_names



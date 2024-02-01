import json
from pathlib import Path
from typing import Tuple, Optional

import pandas as pd
from tqdm import tqdm


def load_data(meetings_dir: str, session_query: Optional[str] = None,
              drop_close_talk: bool = True
              ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load all meetings from the meetings dir

    Args:
        meetings_dir: directory containing meetings.
            Example: project_root/artifacts/meeting_data/dev_set/240121_dev/MTG/
        session_query: a query string to filter the sessions (optional)
            When submitting results, this should be None so no filtering occurs.
        drop_close_talk: whether to drop close-talk devices (optional)
            Close-talk must not be used during inference. However, this can be used as supervision
            signal during training.
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
        if drop_close_talk:
            # drop close-talk devices
            devices_df = devices_df[~devices_df['device_name'].str.startswith('CT')].copy()
        devices_df['meeting_id'] = meeting_subdir.name
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
        all_session_df.query(session_query, inplace=True)

    return all_session_df, all_gt_utt_df, all_metadata_df

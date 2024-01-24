from typing import List, Dict

import pandas as pd
import json
import os


def write_transcript_to_stm(out_dir, attributed_segments_df: pd.DataFrame, tn, session_id: str,
                            filename: str = 'hyp.stm'):
    """
    Save a session's speaker attributed transcription into stm files.

    Args:
        out_dir: the outputs per module are saved to out_dir/{module_name}/{session_id}.
        attributed_segments_df: dataframe of speaker attributed transcribed segments for the given session.
        tn: text normalizer.
        session_id: session name
        filename: the file name to save. Should be hyp.stm for hypothesis
            and ref.stm for reference.

    Returns:
        path to saved stm.
    """
    # assert attributed_segments_df.session_id.nunique() <= 1, 'no cross-session information is permitted'

    filepath = os.path.join(out_dir, 'tcpwer', session_id, filename)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    channel = 1
    with open(filepath, 'w') as f:
        for entry in range(len(attributed_segments_df)):
            speaker_id = attributed_segments_df.iloc[entry]['speaker_id']
            start_time = attributed_segments_df.iloc[entry]['start_time']
            end_time = attributed_segments_df.iloc[entry]['end_time']
            text = tn(attributed_segments_df.iloc[entry]['text'])
            f.write(f'{session_id} {channel} {speaker_id} {start_time} {end_time} {text}\n')

    return filepath


def calc_tcpwer(out_dir: str, hyp_stm_path: str, session_id: str, gt_utt_df: pd.DataFrame, tn,
                collar: float) -> pd.Series:
    """
    Calculates tcpWER for the given session using meeteval dedicated API and saves the error
    information to .json.
    Text normalization is applied to both hypothesis and reference.

    Args:
        out_dir: the directory to save intermediate files to.
        attributed_segments_df: dataframe of speaker attributed transcribed segments for the given session.
        gt_utt_df: dataframe of ground truth utterances for the given session.
        tn: text normalizer TODO: which one exactly?
        collar: tolerance of tcpWER to temporal misalignment between hypothesis and reference.
    Returns:
        session_res: pd.Series with keys -
            'session_id'
            'hyp_file_name': absolute path to .stm files that contain hypothesis of pipeline per session.
            'ref_file_name': absolute path to .stm files that contain ground truth per session.
            'tcp_wer': tcpWER.
            ... other useful tcp_wer keys (see keys below)
    """

    assert os.path.join(out_dir, 'tcpwer', session_id, 'hyp.stm') == hyp_stm_path
    assert gt_utt_df.meeting_id.nunique() <= 1, 'GT should come from a single session'

    ref_stm_path = write_transcript_to_stm(out_dir, gt_utt_df, tn, session_id, filename='ref.stm')

    stm_res = pd.Series(
        {'session_id': session_id, 'hyp_file_name': hyp_stm_path, 'ref_file_name': ref_stm_path})

    def calc_session_wer(session: pd.Series):
        os.system(f"python -m meeteval.wer tcpwer -h {session.hyp_file_name} -r "
                  f"{session.ref_file_name} "
                  f"--collar {collar}")

        with open(os.path.splitext(session.hyp_file_name)[0] + '_tcpwer.json', "r") as read_file:
            data = json.load(read_file)
            keys = ['error_rate', 'errors', 'length', 'insertions', 'deletions', 'substitutions',
                    'missed_speaker', 'falarm_speaker', 'scored_speaker', 'assignment']

            return pd.Series({key: data[key] for key in keys}).rename({'error_rate': 'tcp_wer'})

    wer_res = calc_session_wer(stm_res)
    session_res = pd.concat([stm_res, wer_res], axis=0)

    return session_res

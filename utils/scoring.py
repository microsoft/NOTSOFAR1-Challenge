from pathlib import Path
from dataclasses import dataclass

import pandas as pd
import json
import os
import sys

import meeteval
from meeteval.viz.visualize import AlignmentVisualization

from utils.logging_def import get_logger

_LOG = get_logger('wer')


@dataclass
class ScoringCfg:
    # If True, saves reference - hypothesis visualizations (self-contained html)
    save_visualizations: bool = False


def write_transcript_to_stm(out_dir, attributed_segments_df: pd.DataFrame, tn, session_id: str,
                            filename):
    """
    Save a session's speaker attributed transcription into stm files.

    Args:
        out_dir: the outputs per module are saved to out_dir/{module_name}/{session_id}.
        attributed_segments_df: dataframe of speaker attributed transcribed segments for the given session.
        tn: text normalizer.
        session_id: session name
        filename: the file name to save. Should be, e.g., tcpwer_hyp.stm for hypothesis
            and ref.stm for reference.

    Returns:
        path to saved stm.
    """
    if 'session_id' in attributed_segments_df:
        assert attributed_segments_df.session_id.nunique() <= 1, 'no cross-session information is permitted'

    filepath = Path(out_dir) / 'wer' / session_id / filename
    filepath.parent.mkdir(parents=True, exist_ok=True)
    channel = 1  # ignored by MeetEval

    with filepath.open('w', encoding='utf-8') as f:
        # utf-8 encoding to handle non-ascii characters that may be output by some ASRs
        for entry in range(len(attributed_segments_df)):
            stream_id = attributed_segments_df.iloc[entry]['stream_id']
            start_time = attributed_segments_df.iloc[entry]['start_time']
            end_time = attributed_segments_df.iloc[entry]['end_time']
            text = tn(attributed_segments_df.iloc[entry]['text'])
            f.write(f'{session_id} {channel} {stream_id} {start_time} {end_time} {text}\n')

    return str(filepath)


def calc_wer(out_dir: str, tcp_wer_hyp_stm: str, tcorc_wer_hyp_stm: str,session_id: str,
             gt_utt_df: pd.DataFrame, tn, collar: float, save_visualizations: bool) -> pd.Series:
    """
    Calculates tcpWER for the given session using meeteval dedicated API and saves the error
    information to .json.
    Text normalization is applied to both hypothesis and reference.

    Args:
        out_dir: the directory to save intermediate files to.
        tcp_wer_hyp_stm: path to hypothesis .stm file for tcpWER.
        tcorc_wer_hyp_stm: path to hypothesis .stm file for tcorcWER.
        session_id: session name
        gt_utt_df: dataframe of ground truth utterances for the given session.
        tn: text normalizer
        collar: tolerance of tcpWER to temporal misalignment between hypothesis and reference.
        save_visualizations: if True, save html visualizations of alignment between hyp and ref.
    Returns:
        session_res: pd.Series with keys -
            'session_id'
            'hyp_file_name': absolute path to .stm files that contain hypothesis of pipeline per session.
            'ref_file_name': absolute path to .stm files that contain ground truth per session.
            'tcp_wer': tcpWER.
            ... other useful tcp_wer keys (see keys below)
    """
    assert gt_utt_df.meeting_id.nunique() <= 1, 'GT should come from a single session'

    df = gt_utt_df.copy()
    df['stream_id'] = df['speaker_id']
    ref_stm_path = write_transcript_to_stm(out_dir, df, tn, session_id, filename='ref.stm')

    stm_res = pd.Series(
        {'session_id': session_id, 'tcp_wer_hyp_stm': tcp_wer_hyp_stm,
         'tcorc_wer_hyp_stm': tcorc_wer_hyp_stm, 'ref_stm': ref_stm_path})

    
    def save_wer_visualization(session: pd.Series):
        ref = meeteval.io.load(session.ref_stm).groupby('filename')
        hyp = meeteval.io.load(session.tcp_wer_hyp_stm).groupby('filename')
        assert len(ref) == 1, 'Multiple meetings in a ref file?'
        assert len(hyp) == 1, 'Multiple meetings in a hyp file?'
        assert list(ref.keys())[0] == list(hyp.keys())[0]
        
        meeting_name = list(ref.keys())[0]        
        av = AlignmentVisualization(ref[meeting_name], hyp[meeting_name], alignment='tcp')        
        # Create standalone HTML file
        av.dump(os.path.join(os.path.split(session.tcp_wer_hyp_stm)[0], 'viz.html'))  
        
    
    def calc_session_tcp_wer(session: pd.Series):
        os.system(f"{sys.executable} -m meeteval.wer tcpwer "
                  f"-h {session.tcp_wer_hyp_stm} "
                  f"-r {session.ref_stm} "
                  f"--collar {collar}")

        with (open(os.path.splitext(session.tcp_wer_hyp_stm)[0] + '_tcpwer.json', "r") as read_file):
            data = json.load(read_file)
            keys = ['error_rate', 'errors', 'length', 'insertions', 'deletions', 'substitutions',
                    'missed_speaker', 'falarm_speaker', 'scored_speaker', 'assignment']

            return pd.Series({('tcp_' + key): data[key] for key in keys}
                             ).rename({'tcp_error_rate': 'tcp_wer'})


    def calc_session_tcorc_wer(session: pd.Series):
        os.system(f"{sys.executable} -m meeteval.wer tcorcwer "
                  f"-h {session.tcorc_wer_hyp_stm} "
                  f"-r {session.ref_stm} " 
                  f"--collar {collar}")

        with (open(os.path.splitext(session.tcorc_wer_hyp_stm)[0] + '_tcorcwer.json', "r") as read_file):
            data = json.load(read_file)
            keys = ['error_rate', 'errors', 'length', 'insertions', 'deletions', 'substitutions',
                    'assignment']

            return pd.Series({('tcorc_'+key): data[key] for key in keys}
                             ).rename({'tcorc_error_rate': 'tcorc_wer'})

    tcp_wer_res = calc_session_tcp_wer(stm_res)
    tcorc_wer_res = calc_session_tcorc_wer(stm_res)
    if save_visualizations:
        save_wer_visualization(stm_res)

    session_res = pd.concat([stm_res, tcp_wer_res, tcorc_wer_res], axis=0)

    _LOG.info(f"tcp_wer = {session_res.tcp_wer:.4f}, tcorc_wer = {session_res.tcorc_wer:.4f} "
              f"for session {session_res.session_id}")

    return session_res

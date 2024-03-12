import decimal
from functools import partial
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Callable
import os

import pandas as pd
import meeteval
import meeteval.io.chime7
from meeteval.io.seglst import SegLstSegment
from meeteval.viz.visualize import AlignmentVisualization

from utils.logging_def import get_logger
from utils.text_norm_whisper_like import get_txt_norm

_LOG = get_logger('wer')


@dataclass
class ScoringCfg:
    # If True, saves reference - hypothesis visualizations (self-contained html)
    save_visualizations: bool = False


def df_to_seglst(df):
    return meeteval.io.SegLST([
        SegLstSegment(
            session_id=row.session_id,
            start_time=decimal.Decimal(row.start_time),
            end_time=decimal.Decimal(row.end_time),
            words=row.text,
            speaker=row.speaker_id,
        )
        for row in df.itertuples()
    ])


def normalize_segment(segment: SegLstSegment, tn):
    words = segment["words"]
    words = tn(words)
    segment["words"] = words
    return segment


def calc_wer(out_dir: str,
             tcp_wer_hyp_json: str | List[Dict],
             tcorc_wer_hyp_json: str | List[Dict],
             gt_utt_df: pd.DataFrame, tn: str | Callable = 'chime8',
             collar: float = 5, save_visualizations: bool = False) -> pd.DataFrame:
    """
    Calculates tcpWER and tcorcWER for each session in hypothesis files using meeteval, and saves the error
    information to .json.
    Text normalization is applied to both hypothesis and reference.

    Args:
        out_dir: the directory to save the ref.json reference transcript to (extracted from gt_utt_df).
        tcp_wer_hyp_json: path to hypothesis .json file for tcpWER, or json structure.
        tcorc_wer_hyp_json: path to hypothesis .json file for tcorcWER, or json structure.
        gt_utt_df: dataframe of ground truth utterances. must include the sessions in the hypothesis files.
            see load_data() function.
        tn: text normalizer
        collar: tolerance of tcpWER to temporal misalignment between hypothesis and reference.
        save_visualizations: if True, save html visualizations of alignment between hyp and ref.
    Returns:
        wer_df: pd.DataFrame with columns -
            'session_id' - same as in hypothesis files
            'tcp_wer': tcpWER
            'tcorc_wer': tcorcWER
            ... intermediate tcpWER/tcorcWER fields such as insertions/deletions. see in code.
    """
    # json to SegLST structure (Segment-wise Long-form Speech Transcription annotation)
    to_seglst = lambda x: meeteval.io.chime7.json_to_stm(x, None).to_seglst() if isinstance(x, list) \
        else meeteval.io.load(Path(x))
    tcp_hyp_seglst = to_seglst(tcp_wer_hyp_json)
    tcorc_hyp_seglst = to_seglst(tcorc_wer_hyp_json)

    # map session_id to meetind_id and join with gt_utt_df to include GT utterances for each session.
    # since every meeting contributes several sessions, a meeting's GT will be repeated for every session.
    sess2meet_id = tcp_hyp_seglst.groupby('session_id').keys()
    sess2meet_id = pd.DataFrame(sess2meet_id, columns=['session_id'])
    sess2meet_id['meeting_id'] = sess2meet_id['session_id'].str.extract(r'(MTG_\d+)')
    joined_df = pd.merge(sess2meet_id, gt_utt_df, on='meeting_id', how='left')
    ref_seglst = df_to_seglst(joined_df)

    if isinstance(tn, str):
        tn = get_txt_norm(tn)
    # normalization should be idempotent so a second normalization will not change the result
    tcp_hyp_seglst = tcp_hyp_seglst.map(partial(normalize_segment, tn=tn))
    tcorc_hyp_seglst = tcorc_hyp_seglst.map(partial(normalize_segment, tn=tn))
    ref_seglst = ref_seglst.map(partial(normalize_segment, tn=tn))

    ref_file_path = Path(out_dir) / 'ref.json'
    ref_file_path.parent.mkdir(parents=True, exist_ok=True)
    ref_seglst.dump(ref_file_path)

    def save_wer_visualization(ref, hyp):
        ref = ref.groupby('session_id')
        hyp = hyp.groupby('session_id')
        assert len(ref) == 1 and len(hyp) == 1, 'expecting one session for visualization'
        assert list(ref.keys())[0] == list(hyp.keys())[0]

        meeting_name = list(ref.keys())[0]
        av = AlignmentVisualization(ref[meeting_name], hyp[meeting_name], alignment='tcp')
        # Create standalone HTML file
        av.dump(os.path.join(out_dir, 'viz.html'))

    def calc_session_tcp_wer(ref, hyp):
        res = meeteval.wer.tcpwer(reference=ref, hypothesis=hyp, collar=collar)

        res_df = pd.DataFrame.from_dict(res, orient='index').reset_index(names='session_id')
        keys = ['error_rate', 'errors', 'length', 'insertions', 'deletions', 'substitutions',
                'missed_speaker', 'falarm_speaker', 'scored_speaker', 'assignment']
        return (res_df[['session_id'] + keys]
                .rename(columns={k: 'tcp_' + k for k in keys})
                .rename(columns={'tcp_error_rate': 'tcp_wer'}))

    def calc_session_tcorc_wer(ref, hyp):
        res = meeteval.wer.tcorcwer(reference=ref, hypothesis=hyp, collar=collar)

        res_df = pd.DataFrame.from_dict(res, orient='index').reset_index(names='session_id')
        keys = ['error_rate', 'errors', 'length', 'insertions', 'deletions', 'substitutions', 'assignment']
        return (res_df[['session_id'] + keys]
                .rename(columns={k: 'tcorc_' + k for k in keys})
                .rename(columns={'tcorc_error_rate': 'tcorc_wer'}))

    tcp_wer_res = calc_session_tcp_wer(ref_seglst, tcp_hyp_seglst)
    tcorc_wer_res = calc_session_tcorc_wer(ref_seglst, tcorc_hyp_seglst)
    if save_visualizations:
        save_wer_visualization(ref_seglst, tcp_hyp_seglst)

    wer_df = pd.concat([tcp_wer_res, tcorc_wer_res.drop(columns='session_id')], axis=1)

    if isinstance(tcp_wer_hyp_json, str | Path):
        wer_df['tcp_wer_hyp_json'] = tcp_wer_hyp_json
    if isinstance(tcorc_wer_hyp_json, str | Path):
        wer_df['tcorc_wer_hyp_json'] = tcorc_wer_hyp_json

    _LOG.info('Done calculating WER')
    _LOG.info(f"\n{wer_df[['session_id', 'tcp_wer', 'tcorc_wer']]}")

    return wer_df


def write_submission_jsons(out_dir: str, hyp_jsons_df: pd.DataFrame):
    """
    Merges the per-session jsons in hyp_jsons_df and writes them under the appropriate track folder
    in out_dir.
    The resulting jsons can be used for submission.
    """
    # close-talk is not supposed to be used for scoring
    hyp_jsons_df = hyp_jsons_df[~hyp_jsons_df.is_close_talk]

    def write_json(files, file_name, is_mc):
        seglst = []
        for f in files:
            data = meeteval.io.load(f)
            seglst.extend(data)
        seglst = meeteval.io.SegLST(seglst)
        track = 'multichannel' if is_mc else 'singlechannel'
        filepath = Path(out_dir) / 'wer' / track / file_name
        seglst.dump(filepath)
        _LOG.info(f'Wrote hypothesis transcript for submission: {filepath}')

    mc_hyps = hyp_jsons_df[hyp_jsons_df.is_mc]
    sc_hyps = hyp_jsons_df[~hyp_jsons_df.is_mc]

    if len(mc_hyps) > 0:
        write_json(mc_hyps.tcp_wer_hyp_json, 'tcp_wer_hyp.json', is_mc=True)
        write_json(mc_hyps.tcorc_wer_hyp_json, 'tc_orc_wer_hyp.json', is_mc=True)

    if len(sc_hyps) > 0:
        write_json(sc_hyps.tcp_wer_hyp_json, 'tcp_wer_hyp.json', is_mc=False)
        write_json(sc_hyps.tcorc_wer_hyp_json, 'tc_orc_wer_hyp.json', is_mc=False)


from dataclasses import field, dataclass
from functools import partial
from pathlib import Path
from typing import Optional

import tqdm
import pandas as pd

from asr.asr import asr_inference, WhisperAsrCfg
from css.css import css_inference, CssCfg
from diarization.diarization import diarization_inference
from diarization.diarization_common import DiarizationCfg
from inference_pipeline.load_meeting_data import load_data
from utils.logging_def import get_logger
from utils.scoring import ScoringCfg, calc_wer, df_to_seglst, normalize_segment, write_submission_jsons

_LOG = get_logger('inference')


@dataclass
class InferenceCfg:
    css: CssCfg = field(default_factory=CssCfg)
    asr: WhisperAsrCfg = field(default_factory=WhisperAsrCfg)
    diarization: DiarizationCfg = field(default_factory=DiarizationCfg)
    scoring: ScoringCfg = field(default_factory=ScoringCfg)
    # Optional: Query to filter all_session_df. Useful for debugging. Must be None during full evaluation.
    session_query: Optional[str] = None


@dataclass
class FetchFromCacheCfg:
    css: bool = False
    asr: bool = False
    diarization: bool = False


def inference_pipeline(meetings_dir: str, models_dir: str, out_dir: str, cfg: InferenceCfg,
                       cache: FetchFromCacheCfg):
    """
    Run the inference pipeline on sessions loaded from meetings_dir.
    
    Args:
        meetings_dir: directory with meeting data. 
            example: project_root/artifacts/meeting_data/dev_set/240121_dev/MTG/
        models_dir: directory with CSS models.
            example: project_root/artifacts/css_models/
        out_dir: modules will write their outputs here.
        cfg: config per module.
        cache: basic cache mechanism to re-use results per module. Off by default.
            Note: use at your own risk. If you modify code or config, make sure to delete the cache 
            or set to False.
    """
    # Load all meetings from the meetings dir
    _LOG.info(f'loading meetings from: {meetings_dir}')
    all_session_df, all_gt_utt_df, all_gt_metadata_df = load_data(meetings_dir, cfg.session_query)

    wer_dfs, hyp_jsons = [], []
    # Process each session independently. (Cross-session information is not permitted)
    for _, session in tqdm.tqdm(all_session_df.iterrows(), desc='processing sessions'):
        _LOG.info(f'Processing session: {session.session_id}')

        # Front-end: split session into enhanced streams without overlap speech
        session: pd.Series = css_inference(out_dir, models_dir, session, cfg.css, cache.css)

        # Run ASR on each stream and return transcribed segments
        segments_df: pd.DataFrame = asr_inference(out_dir, session, cfg.asr, cache.asr)

        # Return speaker attributed segments (re-segmentation can occur)
        attributed_segments_df: pd.DataFrame = (
            diarization_inference(out_dir, segments_df, cfg.diarization, cache.diarization))

        # Write hypothesis transcription to: outdir / wer / {multi|single}channel / session_id / *.json
        # These will be merged into one json per track (mc/sc) for submission below.
        hyp_paths: pd.Series = write_hypothesis_jsons(
            out_dir, session, attributed_segments_df, cfg.asr.text_normalizer())
        hyp_jsons.append(hyp_paths)

        # Calculate session WER if GT is available
        if all_gt_utt_df is not None:
            # Rules: WER metric, arguments (collar), and text normalizer must remain unchanged
            calc_wer_out = Path(out_dir) / 'wer' / session.session_id
            session_wer: pd.DataFrame = calc_wer(
                calc_wer_out,
                hyp_paths.tcp_wer_hyp_json,
                hyp_paths.tcorc_wer_hyp_json,
                all_gt_utt_df,
                cfg.asr.text_normalizer(),
                collar=5, save_visualizations=cfg.scoring.save_visualizations)
            wer_dfs.append(session_wer)

    # To submit results to one of the tracks, upload the tcp_wer_hyp.json and tc_orc_wer_hyp.json located in:
    # outdir/wer/{singlechannel | multichannel}/
    hyp_jsons_df = pd.DataFrame(hyp_jsons)
    write_submission_jsons(out_dir, hyp_jsons_df)

    if wer_dfs:  # GT available
        all_session_wer_df = pd.concat(wer_dfs, ignore_index=True)
        _LOG.info(f'Results:\n{all_session_wer_df}')
        _LOG.info(f'mean tcp_wer = {all_session_wer_df["tcp_wer"].mean()}')
        _LOG.info(f'mean tcorc_wer = {all_session_wer_df["tcorc_wer"].mean()}')

        # write session level results into a file
        exp_id = "_".join(['css', cfg.asr.model_name, cfg.diarization.method])
        result_file = Path(out_dir) / "wer" / f"{exp_id}_results.csv"
        result_file.parent.mkdir(parents=True, exist_ok=True)
        all_session_wer_df.to_csv(result_file, sep="\t")
        _LOG.info(f"Wrote full results to: {result_file}")
        # TODO confidence intervals, WER per meta-data


def write_hypothesis_jsons(out_dir, session: pd.Series,
                          attributed_segments_df: pd.DataFrame,
                          text_normalizer):
    """
    Write hypothesis transcripts for session, to be used for tcpwer and tcorwer metrics.
    """

    _LOG.info(f'Writing hypothesis transcripts for session {session.session_id}')

    def write_json(df, filename):
        filepath = Path(out_dir) / 'wer' / session.session_id / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)
        seglst = df_to_seglst(df)
        seglst = seglst.map(partial(normalize_segment, tn=text_normalizer))
        seglst.dump(filepath)
        _LOG.info(f'Wrote {filepath}')
        return filepath

    # I. hyp file for tcpWER
    tcp_wer_hyp_json = write_json(attributed_segments_df, 'tcp_wer_hyp.json')

    # II. hyp file for tcORC-WER, a supplementary metric for analysis.
    # meeteval.wer.tcorcwer requires a stream ID, which depends on the system.
    # Overlapped words should go into different streams, or appear in one stream while respecting the order
    # in reference. See https://github.com/fgnt/meeteval.
    # In NOTSOFAR we define the streams as the outputs of CSS (continuous speech separation).
    # If your system does not have CSS you need to define the streams differently.
    # For example: for end-to-end multi-talker ASR you might use a single stream.
    # Alternatively, you could use the predicted speaker ID as the stream ID.

    # The wav_file_name column of attributed_segments_df indicates the source CSS stream.
    # Note that the diarization module ensures the words within each segment have a consistent channel.
    df_tcorc = attributed_segments_df.copy()
    # Use factorize to map each unique wav_file_name to an index.
    # meeteval.wer.tcorcwer treats speaker_id field as stream id.
    df_tcorc['speaker_id'], uniques = pd.factorize(df_tcorc['wav_file_name'], sort=True)
    _LOG.debug(f'Found {len(uniques)} streams for tc_orc_wer_hyp.stm')
    tcorc_wer_hyp_json = write_json(df_tcorc, 'tc_orc_wer_hyp.json')

    return pd.Series({
        'session_id': session.session_id,
        'tcp_wer_hyp_json': tcp_wer_hyp_json,
        'tcorc_wer_hyp_json': tcorc_wer_hyp_json,
        'is_mc': session.is_mc,
        'is_close_talk': session.is_close_talk,
    })

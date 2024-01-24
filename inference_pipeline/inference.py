import logging
import pprint
from dataclasses import field, dataclass
from typing import Optional
import pandas as pd
import os
from pathlib import Path
from asr.asr import asr_inference, WhisperAsrCfg
from css.css import css_inference, CssCfg
from diarization.diarization import diarization_inference
from diarization.diarization_common import DiarizationCfg
from inference_pipeline.load_meeting_data import load_data
from utils.conf import get_conf
from utils.azure_storage import download_meeting_subset, download_models
from utils.scoring import calc_tcpwer, write_transcript_to_stm
import tqdm


_LOG = logging.getLogger('inference_pipeline')


@dataclass
class InferenceCfg:
    css: CssCfg = field(default_factory=CssCfg)
    asr: WhisperAsrCfg = field(default_factory=WhisperAsrCfg)
    diarization: DiarizationCfg = field(default_factory=DiarizationCfg)
    # Optional: Query to filter all_session_df. Useful for debugging. Must be None during full evaluation.
    session_query: Optional[str] = None


class FetchFromCacheCfg:
    css: bool = False
    asr: bool = False
    diarization: bool = False


def inference_pipeline(meetings_dir: str, models_dir: str, out_dir: str, cfg: InferenceCfg,
                       cache: FetchFromCacheCfg):
    f"""
    Run the inference pipeline on all sessions in the meetings_dir.
    
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
    all_session_df, all_gt_utt_df, all_gt_metadata_df = load_data(meetings_dir, cfg.session_query)

    # Process each session independently. (Cross-session information is not permitted)
    wer_series_list = []
    for session_name, session in tqdm.tqdm(all_session_df.iterrows()):

        # Front-end: split session into enhanced streams without overlap speech
        session: pd.Series = css_inference(out_dir, models_dir, session, cfg.css, cache.css)

        # Run ASR on each stream and return transcribed segments
        segments_df: pd.DataFrame = asr_inference(out_dir, session, cfg.asr, cache.asr)

        # Return speaker attributed segments (re-segmentation can occur)
        attributed_segments_df: pd.DataFrame = diarization_inference(out_dir,
                                                                     segments_df,
                                                                     cfg.diarization,
                                                                     cache.diarization)

        # Write hypothesis transcription to: outdir / tcpwer / session_id / hyp.stm
        # To submit your system for evaluation, send us the contents of: outdir / tcpwer
        hyp_stm = write_transcript_to_stm(out_dir, attributed_segments_df, cfg.asr.text_normalizer(),
                                          session.session_id)

        # Rules: WER metric, arguments (collar), and text normalizer must remain unchanged
        session_wer: pd.Series = calc_tcpwer(out_dir,
                                             hyp_stm,
                                             session.session_id,
                                             get_session_gt(session, all_gt_utt_df),
                                             cfg.asr.text_normalizer(), collar=5)
        wer_series_list.append(session_wer)
        print(f"tcp_wer = {session_wer.tcp_wer:.4f} for session {session_wer.session_id}")


    all_session_wer_df = pd.DataFrame(wer_series_list)    
    print(all_session_wer_df.drop(["hyp_file_name", "ref_file_name", "assignment"], axis=1))
    print(f'mean tcp_wer = {all_session_wer_df["tcp_wer"].mean()}')

    # write session level results into a file
    exp_id = "_".join([os.path.basename(cfg.css.checkpoint_sc), 
                       cfg.asr.model_name,
                       cfg.diarization.method])
    os.makedirs(os.path.join(out_dir, "results"), exist_ok=True)
    result_file = os.path.join(out_dir, "results", exp_id+".tsv")
    print(f"Dump results to {result_file}")
    all_session_wer_df.to_csv(result_file, sep="\t")

    # TODO SAgWER
    # TODO confidence intervals, WER per meta-data


def get_session_gt(session: pd.Series, all_gt_utt_df: pd.DataFrame):
    return all_gt_utt_df[all_gt_utt_df['meeting_id'] == session.meeting_id]


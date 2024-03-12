import os
import json
import glob
import shutil

import pandas as pd
import numpy as np
from omegaconf import OmegaConf

from nemo.collections.asr.models import ClusteringDiarizer
from nemo.collections.asr.models.msdd_models import NeuralDiarizer

from utils.audio_utils import read_wav, write_wav
from diarization.diarization_common import prepare_diarized_data_frame, DiarizationCfg
from utils.logging_def import get_logger

_LOG = get_logger('time_based_diarization')


def run_nemo_diarization(audio_files: list, session_output_dir: str, cfg: DiarizationCfg, vad_time_resolution: float=0.01):
    """
    Run the diarization recipes from the NeMo toolkit. Two recipes can be used: NMESC and NMESC + MSDD. 
    
    After diarization, represent the diarization results as a CxSxT tensor, where C is the number of unmixed
    channels, S is the number of diarized speakers, and T is the number of frames (defined by vad_time_resolution). 
    """
    os.makedirs(session_output_dir, exist_ok=True)
    num_audio_files = len(audio_files)
    if num_audio_files > 1:
        # if there are more than one unmixed channels, concatenate them and diarize.
        # Note that the time order information of the speech segments may not be proporly used in the diarization.
        wavs = [read_wav(audio_file, normalize=True) for audio_file in audio_files]
        audio_file_to_diarize = os.path.join(session_output_dir, "concatenated.wav")
        write_wav(audio_file_to_diarize, np.hstack(wavs))
    else:
        audio_file_to_diarize = audio_files[0]

    manifest = {"audio_filepath": audio_file_to_diarize,
                    "offset": 0,
                    "duration": None, 
                    "label": "infer",
                    "text": "-",
                    "num_speakers": None,
                    "rttm_filepath": None,
                    "uem_filepath": None,
                    }
    manifest_file = os.path.join(session_output_dir, "manifest.json")
    json.dump(manifest, open(manifest_file, "w"))   # don't use indent
        
    if cfg.method == "nmesc":
        # config_file = "configs/inference/diarization/nemo/diar_infer_general.yaml"
        # config_file = "configs/inference/diarization/nemo/diar_infer_telephonic.yaml"
        config_file = "configs/inference/diarization/nemo/diar_infer_meeting.yaml"
        nemo_conf = OmegaConf.load(config_file)
        nemo_conf.diarizer["manifest_filepath"] = manifest_file
        nemo_conf.diarizer["out_dir"] = session_output_dir
        nemo_conf.diarizer["vad"]["model_path"] = cfg.vad_model_name
        nemo_conf.diarizer["speaker_embeddings"]["model_path"] = cfg.embedding_model_name

        sd_model = ClusteringDiarizer(cfg=nemo_conf).to(nemo_conf.device)
        sd_model.diarize()
        
    elif cfg.method == "nmesc_msdd":
        # config_file = "configs/inference/diarization/nemo/diar_infer_general.yaml"
        config_file = "configs/inference/diarization/nemo/diar_infer_telephonic.yaml"   # so far only this config works with MSDD
        # config_file = "configs/inference/diarization/nemo/diar_infer_meeting.yaml"
        nemo_conf = OmegaConf.load(config_file)
        nemo_conf.diarizer["manifest_filepath"] = manifest_file
        nemo_conf.diarizer["out_dir"] = session_output_dir
        nemo_conf.diarizer["vad"]["model_path"] = cfg.vad_model_name
        nemo_conf.diarizer["speaker_embeddings"]["model_path"] = cfg.embedding_model_name
        nemo_conf.diarizer["msdd_model"]["model_path"] = cfg.msdd_model_name

        diarizer_model = NeuralDiarizer(cfg=nemo_conf).to(nemo_conf.device)
        diarizer_model.diarize()
        
    else:
        raise ValueError(f"Unknown diarization method {cfg.method}!")
    
    # load diarization results from NeMo
    rttm_file = glob.glob(os.path.join(session_output_dir, "pred_rttms", "*.rttm"))
    if len(rttm_file) == 0:
        raise Exception("Diarization RTTM file is not created successfully!")
    elif len(rttm_file) > 1: 
        raise Exception("More than one RTTM file found, expect only 1.")

    with open(rttm_file[0]) as file:
        sys_rttm = [line.rstrip('\n').split() for line in file]
    diarized_segments = [[float(seg[3]), float(seg[4]), seg[7]] for seg in sys_rttm]
    diarized_spk_uniq = sorted(list(set([seg[-1] for seg in diarized_segments])))
    
    # represent diarization results as a global frame-based speaker VAD matrix. 
    # The size of the speaker VAD matrix is SxT, where S is the number of diarized speakers, 
    # and T0 is the number of frames. In the case of multiple unmixed channels, T0 is the sum
    # of the duration of the unmixed channels. 
    max_time = np.max(np.array([seg[:2] for seg in diarized_segments]))
    max_vad_frame = int(np.ceil(max_time / vad_time_resolution))
    spk_vad = np.zeros((len(diarized_spk_uniq), max_vad_frame))
    for seg in diarized_segments:
        start_frame = int(np.round(seg[0]/vad_time_resolution))
        end_frame = int(np.round((seg[0]+seg[1])/vad_time_resolution))
        spk_idx = diarized_spk_uniq.index(seg[2])
        spk_vad[spk_idx, start_frame: end_frame] = 1
    
    # convert the global speaker VAD matrix to channel based speaker VAD matrices with size 
    # CxSxT, where C is the number of unmixed channels, and T is from the duration of one unmixed channel.
    if num_audio_files > 1:
        # divide the global speaker VAD matrix into channel dependent speaker VAD matrices
        max_channel_vad_frame = int(max_vad_frame / num_audio_files)
        channel_spk_vad = np.zeros((num_audio_files, len(diarized_spk_uniq), max_channel_vad_frame))
        for i in range(num_audio_files):
            tmp_vad = spk_vad[:, i*max_channel_vad_frame:(i+1)*max_channel_vad_frame]
            channel_spk_vad[i, :, :tmp_vad.shape[1]] = tmp_vad
    else:
        channel_spk_vad = spk_vad[np.newaxis]
    
    return channel_spk_vad

    
def assign_words_to_speakers(segments_df: pd.DataFrame, spk_vad: np.array, apply_deduplication: bool, vad_time_resolution: float=0.01) -> pd.DataFrame:
    """
    Given the diarization output and ASR word boundary information, assign an ASR word to the diarized speaker that is the 
    most active during the word's time interval. 
    """
    has_unassigned_word = False
    all_words = []
    for _, seg in segments_df.iterrows():
        # get the unmixed channel id for current segment
        channel_id = seg.wav_file_name_ind

        for i, word in enumerate(seg["word_timing"]):
            start_frame = int(np.round(word[1]/vad_time_resolution))
            end_frame = int(np.round(word[2]/vad_time_resolution))
            end_frame = np.maximum(start_frame+1, end_frame)    # make sure there is at least one frame for each word
            
            word_spk_count = spk_vad[channel_id][:, start_frame: end_frame]
            avg_word_spk_count = np.mean(word_spk_count, axis=1)
            if np.sum(avg_word_spk_count) == 0:  # no valid speaker count from diarization
                all_words.append(word+[channel_id, None])
                has_unassigned_word = True
            else:
                most_prob_spk_idx = np.argmax(avg_word_spk_count)
                all_words.append(word+[channel_id, f"spk{most_prob_spk_idx}"])

    if has_unassigned_word:
        word_middle_times = [np.mean(word[1:3]) for word in all_words if word[-1] is not None]
        word_spk_ids = [word[-1] for word in all_words if word[-1] is not None]
        
        for word in all_words:
            if word[-1] is None:
                # if a word is not assigned a speaker label for some reason, use the speaker label of the nearest word
                word_middle_time = np.mean(word[1:3])
                time_diff = np.abs(word_middle_times - word_middle_time)
                closest_word_idx = np.argmin(time_diff)
                word[-1] = word_spk_ids[closest_word_idx]
                _LOG.info(f"Word ({word[0]}, {word[1]:.2f}, {word[2]:.2f}) borrowed speaker ID ({word[-1]}) from word centered at {word_middle_times[closest_word_idx]:.2f}s. Time diff = {time_diff[closest_word_idx]:.2f}")
    
    diarized_segments_df = prepare_diarized_data_frame(all_words, segments_df, apply_deduplication)
    
    return diarized_segments_df


def time_based_diarization(wav_files_sorted, segments_df, output_dir, cfg):
    """
    Run NeMo diarization recipes. Combine the ASR words boundary information and diarizaiton output
    to add a speaker label to each recognized word. 
    """
    # Step 1. Run NeMo diarization
    channel_spk_vad = run_nemo_diarization(wav_files_sorted, output_dir, cfg)

    # Step 2. Assign ASR words to diarized speakers
    attributed_segments_df = assign_words_to_speakers(segments_df, channel_spk_vad, cfg.apply_deduplication)
    
    return attributed_segments_df

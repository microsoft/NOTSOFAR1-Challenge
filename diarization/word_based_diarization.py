import os
import pandas as pd
import numpy as np
from omegaconf import OmegaConf
import torch
from torch.cuda.amp import autocast

from nemo.collections.asr.models.label_models import EncDecSpeakerLabelModel
from nemo.collections.asr.parts.utils.offline_clustering import NMESC, SpectralClustering, cos_similarity, getCosAffinityMatrix, getAffinityGraphMat

from utils.audio_utils import read_wav
from diarization.diarization import DiarizationCfg
from diarization.diarization_common import prepare_diarized_data_frame, DiarizationCfg


def load_speaker_model(model_name: str, device: str):
    """
    Load speaker embedding model defined in the NeMo toolkit.
    """
    print("Loading pretrained {} model from NGC".format(model_name))
    spk_model = EncDecSpeakerLabelModel.from_pretrained(model_name=model_name, map_location=device)
    spk_model.eval()
    
    return spk_model


def run_clustering(raw_affinity_mat: np.array, max_num_speakers: int=8, max_rp_threshold: float=0.06, sparse_search_volume: int=30):
    """
    Run NMESC using the implementation from NeMo toolkit.
    """
    nmesc = NMESC(
            raw_affinity_mat,
            max_num_speakers=max_num_speakers,
            max_rp_threshold=max_rp_threshold,
            sparse_search_volume=sparse_search_volume,
        )

    est_num_of_spk, p_hat_value = nmesc.forward()
    affinity_mat = getAffinityGraphMat(raw_affinity_mat, p_hat_value)
    n_clusters = int(est_num_of_spk.item())

    spectral_model = SpectralClustering(n_clusters=n_clusters)
    cluster_label = spectral_model.forward(affinity_mat)
    
    return cluster_label
    

def extract_speaker_embedding_for_words(segments_df, wavs, sr, spk_model, min_embedding_windows):
    """
    For each word, use its word boundary information to extract multi-scale speaker embedding vectors.
    """
    wav_duration = wavs[0].size / sr

    all_words = []
    all_word_embeddings = []
    for _, seg in segments_df.iterrows():
        # get the unmixed channel id for current segment
        channel_id = int(os.path.splitext(os.path.basename(seg.wav_file_names))[0][-1])

        for word in seg["word_timing"]:
            start_time = word[1]
            end_time = word[2]
            center_time = (start_time + end_time) / 2
            word_duration = end_time - start_time
            
            # extract multi-scale speaker embedding for the word
            word_embedding = []
            for min_window_size in min_embedding_windows:
                if word_duration < min_window_size:
                    # if the word duration is shorter than the window size, use a window centered at the word.
                    # The window may cover other neighboring words
                    start_time2 = np.maximum(0, center_time - min_window_size/2)
                    end_time2 = np.minimum(wav_duration, center_time + min_window_size/2)
                    start_sample = int(start_time2*sr)
                    end_sample = int(end_time2*sr)
                else:
                    start_sample = int(start_time*sr)
                    end_sample = int(end_time*sr)

                ### TO DO
                ### Use batching to increase speed
                word_wav = wavs[channel_id][start_sample:end_sample]
                word_wav = torch.tensor(word_wav[np.newaxis], dtype=torch.float32).to(spk_model.device)
                word_lens = torch.tensor([word_wav.shape[1]], dtype=torch.int).to(spk_model.device)
                with autocast():
                    _, tmp_embedding = spk_model.forward(input_signal=word_wav, input_signal_length=word_lens)
                word_embedding.append(tmp_embedding.cpu().detach())
            
            all_words.append(word)
            all_word_embeddings.append(torch.vstack(word_embedding))
            
    return all_words, all_word_embeddings
            

def word_based_clustering(audio_files: list, segments_df: pd.DataFrame, cfg: DiarizationCfg):
    """
    Treat each ASR word as a segment and run NMESC for clustering.
    
    Here, we implicitly use ASR as the VAD, and only consider the speech regions that are recognized into
    words. For each word, we create a speech segment using the word's time bounaries (start/end times). 
    These word based speech segments are used as the inputs to clustering.
    
    As a word's duration is usually too short for extracting reliable speaker embeddings, this function
    uses longer windows centered at the word to extract speaker embeddings. 

    Motivate by the multi-scale affinity matrixes proposed in NeMo's diarization recipe, this function
    also supports multi-scale speaker embedding extraction. Set multiple window sizes in cfg.min_embedding_window.
    The affinity matrixes of different scales all have the same weights. 

    Note that in NeMo's recipe, larger scale affinity matrix contains fewer elements and resampling is needed
    to make the affinity matrixes of all scales having the same size. In this function, all affinity matrixes 
    have the same size, i.e. NxN, where N is the number of words. So no resampling is needed.
    """
    # load unmixed waveforms
    wavs = [read_wav(audio_file, normalize=True, return_rate=True) for audio_file in audio_files]
    sr = wavs[0][0]
    wavs = np.vstack([wav[1] for wav in wavs])
    
    # load speaker embedding model
    spk_model = load_speaker_model(cfg.embedding_model_name, device=None)
    
    # extract word-based multi-scale speaker embedding vectors
    all_words, all_word_embeddings = extract_speaker_embedding_for_words(segments_df, wavs, sr, spk_model, cfg.min_embedding_windows)

    # compute affinity matrix for clustering
    all_word_embeddings2 = torch.stack(all_word_embeddings)
    emb_t = all_word_embeddings2.half().to(spk_model.device)
    # compute affinity matrix for each scale
    scale_affinity = [getCosAffinityMatrix(emb_t[:, scale]) 
                      for scale in range(len(cfg.min_embedding_windows))]
    # final affinity matrix is the average of scale-dependent affinity matrices
    affinity = torch.mean(torch.stack(scale_affinity), dim=0)
    
    # run NMESC
    cluster_label = run_clustering(affinity)
    
    # prepare segment data frame
    all_words = [word+[f"spk{spk_idx}"] for word, spk_idx in zip(all_words, cluster_label)]
    diarized_segments_df = prepare_diarized_data_frame(all_words, segments_df)
    
    return diarized_segments_df

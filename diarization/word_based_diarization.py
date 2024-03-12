import os
from typing import Optional
import pandas as pd
import numpy as np
import torch
from torch.cuda.amp import autocast
from tqdm import tqdm

from nemo.collections.asr.models.label_models import EncDecSpeakerLabelModel
from nemo.collections.asr.parts.utils.offline_clustering import NMESC, SpectralClustering, cos_similarity, getCosAffinityMatrix, getAffinityGraphMat

from utils.audio_utils import read_wav
from utils.torch_utils import is_dist_initialized
from diarization.diarization import DiarizationCfg
from diarization.diarization_common import prepare_diarized_data_frame, DiarizationCfg
from utils.logging_def import get_logger

_LOG = get_logger('word_based_diarization')


def load_speaker_model(model_name: str, device: str):
    """
    Load speaker embedding model defined in the NeMo toolkit.
    """
    _LOG.info("Loading pretrained {} model from NGC".format(model_name))
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


def extract_speaker_embedding_for_words(segments_df, wavs, sr, spk_model, min_embedding_windows, max_allowed_word_duration=3):
    """
    For each word, use its word boundary information to extract multi-scale speaker embedding vectors.
    """
    wav_duration = wavs[0].size / sr

    all_words = []
    all_word_embeddings = []
    too_long_words = []

    n_words = sum(len(seg['word_timing']) for _, seg in segments_df.iterrows())
    _segments_df, _ = _fill_dummy_words_for_ddp(segments_df)
    words_processed = 0

    for _, seg in tqdm(_segments_df.iterrows(), desc='extracting speaker embedding for segments', total=len(_segments_df)):
        # get the unmixed channel id for current segment
        channel_id = seg.wav_file_name_ind

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
                with autocast(), torch.no_grad():
                    _, tmp_embedding = spk_model.forward(input_signal=word_wav, input_signal_length=word_lens)
                word_embedding.append(tmp_embedding.cpu().detach())

            words_processed += 1

            if words_processed > n_words:
                # This is a dummy word added for DDP. Skip it.
                continue

            if word_duration > max_allowed_word_duration:
                # Very long word duration is very suspicious and may harm diarization. Ignore them for now.
                # Note that these words will disappear in the final result.
                # To do: find a better way to deal with these words.
                _LOG.info(f"word '{word[0]}' has unreasonablly long duration ({start_time}s, {end_time}s). Skip it in diarization")
                too_long_words.append(word)
                continue

            # append only the real words (do not append dummy words)
            all_words.append(word+[channel_id])
            all_word_embeddings.append(torch.vstack(word_embedding))

    print(f'Done extracting embeddings. {words_processed=}, {len(all_words)=}, {n_words=}', flush=True)
    n_real_words = n_words - len(too_long_words)
    assert len(all_words) == n_real_words, f"Number of words {len(all_words)} != n_real_words {n_real_words}"
    return all_words, all_word_embeddings


def word_based_clustering(audio_files: list, segments_df: pd.DataFrame, cfg: DiarizationCfg,
                          device: Optional[str] = None):
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
    srs, wavs = zip(*[read_wav(audio_file, normalize=True, return_rate=True) for audio_file in audio_files])
    sr = srs[0]
    max_length = max([wav.size for wav in wavs])
    # pad to the maximum length and stack. padding is only relevant to segmented close-talk.
    # CSS always returns equal-length channels.
    wavs = np.vstack(
        [np.pad(wav, (0, max_length - wav.size), 'constant', constant_values=(0, 0)) for wav in
         wavs])

    # load speaker embedding model
    spk_model = load_speaker_model(cfg.embedding_model_name, device=device)

    # extract word-based multi-scale speaker embedding vectors
    all_words, all_word_embeddings = extract_speaker_embedding_for_words(segments_df, wavs, sr, spk_model,
                                                                         cfg.min_embedding_windows,
                                                                         cfg.max_allowed_word_duration)

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
    diarized_segments_df = prepare_diarized_data_frame(all_words, segments_df, cfg.apply_deduplication)

    return diarized_segments_df


def _fill_dummy_words_for_ddp(segments_df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """
    Fill the last segment with dummy words to make the number of words the same across all processes in DDP.

    Returns:
        (a COPY of segments_df with dummy words added to the last segment, number of real words, number of dummies)
    """

    if not is_dist_initialized():
        return segments_df, 0

    n_words = sum(len(seg['word_timing']) for _, seg in segments_df.iterrows())
    max_words = get_max_value(n_words)
    print(f"Number of segments: {len(segments_df)}, Number of words: {n_words}, max_words(in DDP): {max_words}")

    # find first segment with non-empty word_timing
    for i in range(len(segments_df)):
        if len(segments_df.iloc[i]['word_timing']) > 0:
            dummy_word = segments_df.iloc[i]['word_timing'][-1].copy()
            break

    # fill last segment with dummy data
    _segments_df = segments_df.copy()
    n_dummies = max_words - n_words
    for _ in range(n_dummies):
        _segments_df.iloc[-1]['word_timing'].append(dummy_word)

    n_words_with_dummies = sum([len(seg['word_timing']) for _, seg in _segments_df.iterrows()])
    assert n_words_with_dummies == max_words, \
        f"Number of words with dummies {n_words_with_dummies} != max_words {max_words}"
    print(f"Number of words to process (with dummies): {n_words_with_dummies}")

    return _segments_df, n_dummies

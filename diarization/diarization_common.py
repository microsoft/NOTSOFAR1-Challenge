import os
import pandas as pd
import numpy as np
from dataclasses import dataclass, field


# diarization inference configuration
@dataclass
class DiarizationCfg:
    method: str = "nmesc"       # choose from "nmesc", "nmesc_msdd", "word_nmesc", or "skip"
    min_embedding_windows: list = field(default_factory=list)
    max_allowed_word_duration: float = 3    # maximum allowed word duration. If word is longer than this value, ignore it.
    apply_deduplication: bool = True
    embedding_model_name: str = "titanet_large"
    msdd_model_name: str = "diar_msdd_telephonic"
    # vad_model_name: str = "vad_telephony_marblenet"    # 8kHz telephone
    vad_model_name: str = "vad_multilingual_marblenet"   # 16kHz


def merge_words_to_segments_by_spk_change(all_words: list) -> dict:
    if len(all_words) == 0:
        return []
    if len(all_words) == 1:
        return all_words
    
    segments = {"word_timing": [],
                "speaker_id": []}
    seg_start = 0
    for i, word in enumerate(all_words):
        # if speaker ID is changed or channel ID is changed, break. This makes sure that each segment 
        # contains words from a single channel, so the segments will be safe to compute tcorc_wer. 
        if i > 0 and (word[-1] != all_words[seg_start][-1] or word[-2] != all_words[seg_start][-2]):
            seg_words = all_words[seg_start: i]
            segments["word_timing"].append([w[:-1] for w in seg_words])
            segments["speaker_id"].append(seg_words[0][-1])
            seg_start = i
    segments["word_timing"].append([w[:-1] for w in all_words[seg_start:]])
    segments["speaker_id"].append(all_words[seg_start][-1])

    return segments


def compute_overlap_ratio(start1, end1, start2, end2):
    latest_start = max(start1, start2)
    earliest_end = min(end1, end2)
    overlap = earliest_end - latest_start
    
    if overlap < 0:
        return 0  # No overlap
    
    duration1 = end1 - start1
    duration2 = end2 - start2
    longer_duration = max(duration1, duration2)
    
    return overlap / longer_duration


def deduplicate(all_words_sorted, overlap_threshold=0.5):
    all_words_deduplicated = []
    for i, curr_word in enumerate(all_words_sorted):
        if i == 0:
            continue
        prev_word = all_words_sorted[i-1]
        skip_word = False
        if curr_word[0] == prev_word[0] and curr_word[4] == prev_word[4]:
            overlap_ratio = compute_overlap_ratio(curr_word[1], curr_word[2], prev_word[1], prev_word[2])
            if overlap_ratio > overlap_threshold:
                # if identifical words belong to the same speaker and are appearing 
                # in multiple unmixed channels, and they have more than 50% overlapped, 
                # only keep the first one. 
                skip_word = True
        if not skip_word:
            all_words_deduplicated.append(curr_word)
                
    return all_words_deduplicated
    

def prepare_diarized_data_frame(all_words, segments_df, apply_deduplication):
    # cut word sequence into segments according to speaker change
    all_words_sorted = sorted(all_words, key=lambda x:x[2])     # sort words by end time
    if apply_deduplication:
        final_words = deduplicate(all_words_sorted)
    else:
        final_words = all_words_sorted
    segments = merge_words_to_segments_by_spk_change(final_words)
    
    diarized_segments_df = pd.DataFrame(
        {'start_time': [seg[0][1] for seg in segments["word_timing"]],
         'end_time': [seg[-1][2] for seg in segments["word_timing"]],
         'text': ["".join([w[0] for w in seg]) for seg in segments["word_timing"]],
         'word_timing': segments["word_timing"]})
        
    diarized_segments_df['meeting_id'] = segments_df['meeting_id'][0]
    diarized_segments_df['session_id'] = segments_df['session_id'][0]
    
    # assign correct CSS file name to each diarized segment
    stream_id = [seg[0][-1] for seg in diarized_segments_df.word_timing.to_list()]
    diarized_segments_df['wav_file_name'] = segments_df['wav_file_name'].cat.categories[stream_id]

    diarized_segments_df['speaker_id'] = segments["speaker_id"]
    
    return diarized_segments_df

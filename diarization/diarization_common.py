import pandas as pd
from dataclasses import dataclass, field


# diarization inference configuration
@dataclass
class DiarizationCfg:
    method: str = "nmesc"       # choose from "nmesc" and "nmesc_msdd"
    min_embedding_windows: list = field(default_factory=list) 
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
        if i > 0 and word[-1] != all_words[seg_start][-1]:
            seg_words = all_words[seg_start: i]
            segments["word_timing"].append([w[:-1] for w in seg_words])
            segments["speaker_id"].append(seg_words[0][-1])
            seg_start = i
    segments["word_timing"].append([w[:-1] for w in all_words[seg_start:]])
    segments["speaker_id"].append(all_words[seg_start][-1])

    return segments


def prepare_diarized_data_frame(all_words, segments_df):
    # cut word sequence into segments according to speaker change
    segments = merge_words_to_segments_by_spk_change(all_words)
    
    diarized_segments_df = pd.DataFrame(
        {'start_time': [seg[0][1] for seg in segments["word_timing"]],
         'end_time': [seg[-1][2] for seg in segments["word_timing"]],
         'text': ["".join([w[0] for w in seg]) for seg in segments["word_timing"]],
         'word_timing': segments["word_timing"]})
        
    diarized_segments_df['meeting_id'] = segments_df['meeting_id'][0]
    diarized_segments_df['session_id'] = segments_df['session_id'][0]
    diarized_segments_df['wav_file_names'] = segments_df['wav_file_names'][0]
    diarized_segments_df['speaker_id'] = segments["speaker_id"]
    
    return diarized_segments_df

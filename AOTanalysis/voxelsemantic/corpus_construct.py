import os
from AOTaccess.stimulus_info_access import StimuliInfoAccess
from AOTanalysis.bandedRR.construct_features import (
    construct_features_motion_energy,
    construct_features_sbert_embeddings,
)


def construct_AOT_corpus():
    """construct the AOT corpus"""
    aotaccess = StimuliInfoAccess()
    video_indexes = [i for i in range(1, 2179)]
    corpus = []
    for video_index in video_indexes:
        sentence = aotaccess._temp_read_llama_description(video_index)
        corpus.append(sentence)
    embeddings = construct_features_sbert_embeddings(
        video_indexes, duplicate=False, centered=True
    )

    return corpus, embeddings

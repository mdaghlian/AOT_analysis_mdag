import numpy as np
from pathlib import Path
from AOTaccess.stimulus_info_access import StimuliInfoAccess
from AOTaccess.glmsingle_access import GLMSingleAccess
from AOTaccess.expdesign_access import ExpDesignAccess
from copy import deepcopy

# import PCA
from sklearn.decomposition import PCA


def construct_universal_average(feature_name: str):
    """
    Construct universal average feature on the whole dataset on the sample level.

    Parameters:
    feature_name (str): Name of the feature to construct the average for.

    Returns:
    np.ndarray: The universal average feature.
    """
    save_dir = "/tank/shared/2024/visual/AOT/temp/feature_average"
    video_index = [i for i in range(1, 2179)]
    if feature_name == "motion_energy16":
        file_path = Path(save_dir) / "motion_energy_average.npy"
        if file_path.exists():
            average = np.load(file_path)
            print(f"Shape of universal average {feature_name}: {average.shape}")
        else:
            all_features = construct_features_motion_energy(
                video_index, highest_freq=16
            )
            average = np.mean(all_features, axis=0)
            print(f"Shape of universal average {feature_name}: {average.shape}")
            # save the universal average
            np.save(file_path, average)
    elif feature_name == "motion_energy32":
        file_path = Path(save_dir) / "motion_energy_average32.npy"
        if file_path.exists():
            average = np.load(file_path)
            print(f"Shape of universal average {feature_name}: {average.shape}")
        else:
            all_features = construct_features_motion_energy(
                video_index, highest_freq=32
            )
            average = np.mean(all_features, axis=0)
            print(f"Shape of universal average {feature_name}: {average.shape}")
            # save the universal average
            np.save(file_path, average)
    elif feature_name == "semantic_embeddings":
        file_path = Path(save_dir) / "semantic_embeddings_average.npy"
        if file_path.exists():
            average = np.load(file_path)
            print(f"Shape of universal average {feature_name}: {average.shape}")
        else:
            all_features = construct_features_sbert_embeddings(video_index)
            average = np.mean(all_features, axis=0)
            print(f"Shape of universal average {feature_name}: {average.shape}")
            # save the universal average
            np.save(file_path, average)
    else:
        raise ValueError("feature name do not match")

    return average


def flatten_motion_energy_feature(motion_energy_feature):
    """
    Flatten the motion energy feature.

    Parameters:
    motion_energy_feature (np.ndarray): The motion energy feature with shape (frames, filters).

    Returns:
    np.ndarray: The flattened motion energy feature.
    """
    motion_energy_feature = np.array(
        motion_energy_feature
    )  # (frames,filters)   (60,2162)

    feature_dimention = motion_energy_feature.shape[1]
    print(f"Shape of motion energy feature: {motion_energy_feature.shape}")
    timepoints = 60  # motion_energy_feature.shape[0]
    reshape_size = 60 * feature_dimention
    # get first 60 frames for the motion energy feature
    regularized = np.array(
        [motion_energy_feature[t].flatten() for t in range(timepoints)]
    )
    flattend_motion_energy_feature = regularized.reshape(reshape_size)
    print(
        f"Shape of flattened motion energy feature: {flattend_motion_energy_feature.shape}"
    )
    return flattend_motion_energy_feature


def average_motion_energy_feature(motion_energy_feature):
    """
    Average the motion energy feature on the time axis.

    Parameters:
    motion_energy_feature (np.ndarray): The motion energy feature with shape (frames, filters).

    Returns:
    np.ndarray: The averaged motion energy feature.
    """
    motion_energy_feature = np.array(
        motion_energy_feature
    )  # (frames,filters)   (60,2162)
    print(f"Shape of motion energy feature: {motion_energy_feature.shape}")
    avg_motion_energy_feature = np.mean(motion_energy_feature, axis=0)
    print(f"Shape of averaged motion energy feature: {avg_motion_energy_feature.shape}")
    return avg_motion_energy_feature


def construct_features_motion_energy(
    video_index: list, duplicate: bool = True, centered: bool = False, highest_freq=16
):
    """
    Construct features from motion energy.

    Parameters:
    video_index (list): List of video indices.
    duplicate (bool): Whether to duplicate the features.
    centered (bool): Whether to center the features by subtracting the universal average.

    Returns:
    np.ndarray: The constructed motion energy features.
    """
    stim_access = StimuliInfoAccess()
    motion_energy_features = []
    for video_id in video_index:
        motion_energy = stim_access._temp_read_motion_energy_features(
            video_id=video_id, direction="fw", highest_freq=highest_freq
        )
        avg_features = average_motion_energy_feature(motion_energy)  # (2162,)
        motion_energy_features.append(avg_features)
        if duplicate:
            motion_energy_features.append(avg_features)
    if centered:
        if highest_freq == 16:
            universal_average = construct_universal_average("motion_energy16")
        elif highest_freq == 32:
            universal_average = construct_universal_average("motion_energy32")
        motion_energy_features = motion_energy_features - universal_average
    motion_energy_features = np.array(motion_energy_features)
    print(f"Shape of motion energy features: {motion_energy_features.shape}")
    return motion_energy_features


def construct_features_sbert_embeddings(
    video_index: list, duplicate: bool = True, centered: bool = False
):
    """
    Construct features from SBERT embeddings.

    Parameters:
    video_index (list): List of video indices.
    duplicate (bool): Whether to duplicate the features.
    centered (bool): Whether to center the features by subtracting the universal average.

    Returns:
    np.ndarray: The constructed SBERT embeddings.
    """
    stim_access = StimuliInfoAccess()
    sbert_embeddings = []
    for video_id in video_index:
        sbert_embedding = stim_access._temp_read_sbert_embeddings(
            video_id=video_id, direction="fw"
        )
        sbert_embeddings.append(sbert_embedding)
        if duplicate:
            sbert_embeddings.append(sbert_embedding)
    sbert_embeddings = np.array(sbert_embeddings)
    if centered:
        universal_average = construct_universal_average("semantic_embeddings")
        sbert_embeddings = sbert_embeddings - universal_average
    print(f"Shape of sbert embeddings: {sbert_embeddings.shape}")
    return sbert_embeddings


def construct_features_sbert_embeddings_PCA(
    video_index: list, duplicate: bool = True, centered: bool = False
):
    """
    Construct features from SBERT embeddings with PCA.

    Parameters:
    video_index (list): List of video indices.
    duplicate (bool): Whether to duplicate the features.
    centered (bool): Whether to center the features by subtracting the universal average.

    Returns:
    np.ndarray: The constructed SBERT embeddings with PCA.
    """
    stim_access = StimuliInfoAccess()
    sbert_embedding_PCAs = []
    for video_id in video_index:
        sbert_embeddings_PCA = stim_access._temp_read_sbert_embeddings_PCA(
            video_id=video_id, direction="fw"
        )
        sbert_embedding_PCAs.append(sbert_embeddings_PCA)
        if duplicate:
            sbert_embedding_PCAs.append(sbert_embeddings_PCA)
    sbert_embedding_PCAs = np.array(sbert_embedding_PCAs)
    if centered:
        universal_average = construct_universal_average("semantic_embeddings_PCA")
        sbert_embedding_PCAs = sbert_embedding_PCAs - universal_average
    print(f"Shape of sbert embeddings with PCA: {sbert_embedding_PCAs.shape}")
    return sbert_embedding_PCAs


def construct_features_sbert_embeddings_SAE(
    video_index: list, duplicate: bool = True, centered: bool = False
):
    """
    Construct features from SBERT embeddings with SAE.

    Parameters:
    video_index (list): List of video indices.
    duplicate (bool): Whether to duplicate the features.
    centered (bool): Whether to center the features by subtracting the universal average.

    Returns:
    np.ndarray: The constructed SBERT embeddings with SAE.
    """
    stim_access = StimuliInfoAccess()
    sbert_embedding_SAEs = []
    for video_id in video_index:
        sbert_embeddings_SAE = stim_access._temp_read_sbert_embeddings_SAE(
            video_id=video_id, direction="fw"
        )
        sbert_embedding_SAEs.append(sbert_embeddings_SAE)
        if duplicate:
            sbert_embedding_SAEs.append(sbert_embeddings_SAE)
    sbert_embedding_SAEs = np.array(sbert_embedding_SAEs)
    if centered:
        universal_average = construct_universal_average("semantic_embeddings_SAE")
        sbert_embedding_SAEs = sbert_embedding_SAEs - universal_average
    print(f"Shape of sbert embeddings with SAE: {sbert_embedding_SAEs.shape}")
    return sbert_embedding_SAEs


def construct_features_motion_energy_from_subses(
    sub: int,
    ses: int,
    glmtype: str = "TYPED_FITHRF_GLMDENOISE_RR",
    centered: bool = False,
    highest_freq=16,
):
    """
    Construct features from motion energy for a specific subject and session.

    Parameters:
    sub (int): Subject number.
    ses (int): Session number.
    glmtype (str): GLM type.
    centered (bool): Whether to center the features by subtracting the universal average.

    Returns:
    np.ndarray: The constructed motion energy features.
    """
    glm_access = GLMSingleAccess()
    expdesign_access = ExpDesignAccess()
    stim_access = StimuliInfoAccess()
    motion_energy_features = []
    video_indexes = expdesign_access.get_session_video_indexes(sub, ses)
    # make all the video indexes string to int
    video_indexes = [int(video_index) for video_index in video_indexes]
    print(f"Video indexes: {video_indexes}")
    for video_id in video_indexes:
        motion_energy = stim_access._temp_read_motion_energy_features(
            video_id=video_id, direction="fw", highest_freq=highest_freq
        )
        avg_features = average_motion_energy_feature(motion_energy)
        motion_energy_features.append(avg_features)
    motion_energy_features = np.array(motion_energy_features)
    if centered:
        if highest_freq == 16:
            universal_average = construct_universal_average("motion_energy16")
        elif highest_freq == 32:
            universal_average = construct_universal_average("motion_energy32")
        motion_energy_features = motion_energy_features - universal_average
    print(f"Shape of motion energy features: {motion_energy_features.shape}")
    return motion_energy_features


def construct_features_sbert_embeddings_from_subses(
    sub: int,
    ses: int,
    glmtype: str = "TYPED_FITHRF_GLMDENOISE_RR",
    centered: bool = False,
):
    """
    Construct features from SBERT embeddings for a specific subject and session.

    Parameters:
    sub (int): Subject number.
    ses (int): Session number.
    glmtype (str): GLM type.
    centered (bool): Whether to center the features by subtracting the universal average.

    Returns:
    np.ndarray: The constructed SBERT embeddings.
    """
    glm_access = GLMSingleAccess()
    expdesign_access = ExpDesignAccess()
    stim_access = StimuliInfoAccess()
    sbert_embeddings = []
    video_indexes = expdesign_access.get_session_video_indexes(sub, ses)
    # make all the video indexes string to int
    video_indexes = [int(video_index) for video_index in video_indexes]
    print(f"Video indexes: {video_indexes}")
    for video_id in video_indexes:
        sbert_embedding = stim_access._temp_read_sbert_embeddings(
            video_id=video_id, direction="fw"
        )
        sbert_embeddings.append(sbert_embedding)
    sbert_embeddings = np.array(sbert_embeddings)
    if centered:
        universal_average = construct_universal_average("semantic_embeddings")
        sbert_embeddings = sbert_embeddings - universal_average
    print(f"Shape of sbert embeddings: {sbert_embeddings.shape}")
    return sbert_embeddings


def construct_features_sbert_embeddings_PCA_from_subses(
    sub: int,
    ses: int,
    glmtype: str = "TYPED_FITHRF_GLMDENOISE_RR",
    centered: bool = False,
    remove_first_component: bool = False,
):
    """
    Construct features from SBERT embeddings with PCA for a specific subject and session.

    Parameters:
    sub (int): Subject number.
    ses (int): Session number.
    glmtype (str): GLM type.
    centered (bool): Whether to center the features by subtracting the universal average.
    remove_first_component (bool): Whether to remove the first principal component.

    Returns:
    np.ndarray: The constructed SBERT embeddings with PCA.
    """
    glm_access = GLMSingleAccess()
    expdesign_access = ExpDesignAccess()
    stim_access = StimuliInfoAccess()
    sbert_embedding_PCAs = []
    video_indexes = expdesign_access.get_session_video_indexes(sub, ses)
    # make all the video indexes string to int
    video_indexes = [int(video_index) for video_index in video_indexes]
    print(f"Video indexes: {video_indexes}")
    for video_id in video_indexes:
        sbert_embedding_PCA = stim_access._temp_read_sbert_embeddings_PCA(
            video_id=video_id, direction="fw"
        )
        if remove_first_component:
            sbert_embedding_PCA = sbert_embedding_PCA[1:]
        sbert_embedding_PCAs.append(sbert_embedding_PCA)
    sbert_embedding_PCAs = np.array(sbert_embedding_PCAs)
    if centered:
        universal_average = construct_universal_average("semantic_embeddings_PCA")
        sbert_embedding_PCAs = sbert_embedding_PCAs - universal_average
    print(f"Shape of sbert embeddings with PCA: {sbert_embedding_PCAs.shape}")
    return sbert_embedding_PCAs


def construct_features_sbert_embeddings_SAE_from_subses(
    sub: int,
    ses: int,
    glmtype: str = "TYPED_FITHRF_GLMDENOISE_RR",
    centered: bool = False,
):
    """
    Construct features from SBERT embeddings with SAE for a specific subject and session.

    Parameters:
    sub (int): Subject number.
    ses (int): Session number.
    glmtype (str): GLM type.
    centered (bool): Whether to center the features by subtracting the universal average.

    Returns:
    np.ndarray: The constructed SBERT embeddings with SAE.
    """
    glm_access = GLMSingleAccess()
    expdesign_access = ExpDesignAccess()
    stim_access = StimuliInfoAccess()
    sbert_embedding_SAEs = []
    video_indexes = expdesign_access.get_session_video_indexes(sub, ses)
    # make all the video indexes string to int
    video_indexes = [int(video_index) for video_index in video_indexes]
    print(f"Video indexes: {video_indexes}")
    for video_id in video_indexes:
        sbert_embedding_SAE = stim_access._temp_read_sbert_embeddings_SAE(
            video_id=video_id, direction="fw"
        )
        sbert_embedding_SAEs.append(sbert_embedding_SAE)
    sbert_embedding_SAEs = np.array(sbert_embedding_SAEs)
    if centered:
        universal_average = construct_universal_average("semantic_embeddings_SAE")
        sbert_embedding_SAEs = sbert_embedding_SAEs - universal_average
    print(f"Shape of sbert embeddings with SAE: {sbert_embedding_SAEs.shape}")
    return sbert_embedding_SAEs


if __name__ == "__main__":
    sub = 1
    test_video_index = [1, 2, 3]
    motion_energy_features = construct_features_motion_energy(test_video_index)
    sbert_embeddings = construct_features_sbert_embeddings(test_video_index)

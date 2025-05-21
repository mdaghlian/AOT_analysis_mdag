import numpy as np
from pathlib import Path
from AOTaccess.stimulus_info_access import StimuliInfoAccess
from AOTaccess.glmsingle_access import GLMSingleAccess
from AOTaccess.expdesign_access import ExpDesignAccess
from AOTanalysis.bandedRR.utils import split_single_list


def flatten_video_betas(video_betas):
    """
    Flatten the video betas that are in the shape of (t,x,y,z).

    Parameters:
    video_betas (np.ndarray): The video betas with shape (t,x,y,z).

    Returns:
    np.ndarray: The flattened video betas with shape (t, x*y*z).
    """
    video_betas = np.array(video_betas)  # (t,x,y,z)
    print(f"Shape of video betas: {video_betas.shape}")
    timepoints = video_betas.shape[0]
    flattened_volumes = np.array([video_betas[t].flatten() for t in range(timepoints)])
    print(f"Shape of flattened volumes: {flattened_volumes.shape}")
    return flattened_volumes


def recorver_video_betas(video_betas, original_volume_shape):
    """
    Recover the video betas to the original shape.

    Parameters:
    video_betas (np.ndarray): The flattened video betas with shape (t, x*y*z).
    original_volume_shape (tuple): The original shape of the video betas (x,y,z).

    Returns:
    np.ndarray: The recovered video betas with shape (t,x,y,z).
    """
    timepoints = video_betas.shape[0]
    recovered_volumes = np.array(
        [video_betas[t].reshape(original_volume_shape) for t in range(timepoints)]
    )
    print(f"Shape of recovered volumes: {recovered_volumes.shape}")
    return recovered_volumes


def construct_target_data_split_flatten_masked(
    sub: int,
    split_num: int,
    index: int,
    # centered=True,
    zscore=True,
    mask_threshold=0.2,
    randomize=False,
    seed=0,
    direction="fw",
):
    """
    Construct target data by splitting, flattening, and masking video betas.

    Parameters:
    sub (int): Subject number.
    split_num (int): Number of splits.
    index (int): Index of the split to use.
    centered (bool): Whether to center the data by subtracting the mean.
    mask_threshold (float): Threshold for the mask.
    randomize (bool): Whether to randomize the data.
    seed (int): Seed for randomization.
    direction (str): Direction of the video betas.

    Returns:
    tuple: Flattened and masked video betas, and the corresponding video indices.
    """
    glm_access = GLMSingleAccess()
    video_betas = []
    video_index = []
    video_not_found = []
    video_num = 2300  # from 1 to 2300
    for video_id in range(1, video_num + 1):
        betas = glm_access.read_video_betas(
            sub=sub, video_num=video_id, direction=direction, zscore=zscore
        )
        if type(betas) != type(None):
            video_betas.append(betas)
            video_index.append(video_id)
            print(
                f"Loaded betas for video {video_id}"
            )  # two betas for one video, repeated in the experiement
            print(f"Shape of betas: {betas.shape}")
        else:
            print(f"Video {video_id} not found")
            video_not_found.append(video_id)
    # concatenate all the betas
    print("len of found:", len(video_betas))
    print(len(video_index))

    if randomize:
        # randomize the betas and the index in the same way, with a determined seed
        np.random.seed(seed)
        np.random.shuffle(video_betas)
        np.random.seed(seed)
        np.random.shuffle(video_index)

    # split the betas
    video_index_splits = split_single_list(video_index, split_num)
    video_betas_splits = split_single_list(video_betas, split_num)
    video_betas = video_betas_splits[index]
    video_index = video_index_splits[index]

    mask = glm_access.read_R2_mask(sub, ses=1, threshold=mask_threshold)
    mask_reshape = mask.flatten()
    print(f"Shape of mask: {mask.shape}")
    print(f"Shape of mask reshape: {mask_reshape.shape}")
    video_betas = np.concatenate(video_betas, axis=0)
    print(f"Shape of all betas original: {video_betas.shape}")
    video_betas = flatten_video_betas(video_betas)

    video_betas = video_betas[:, mask_reshape]
    print(f"Shape of all betas flattened: {video_betas.shape}")

    # if centered:
    #     video_betas = video_betas - np.mean(video_betas, axis=0)

    return video_betas, video_index


def construct_target_data_from_session_flatten_masked(
    sub: int,
    ses: int,
    # centered=True,
    zscore=True,
    mask_threshold=0.2,
):
    """
    Construct target data from a session by flattening and masking video betas.

    Parameters:
    sub (int): Subject number.
    ses (int): Session number.
    centered (bool): Whether to center the data by subtracting the mean.
    mask_threshold (float): Threshold for the mask.

    Returns:
    np.ndarray: Flattened and masked video betas.
    """
    glmaccess = GLMSingleAccess()
    expdesign_access = ExpDesignAccess()

    session_video_indexes = expdesign_access.get_session_video_indexes(sub, ses)
    session_video_indexes = [int(video_index) for video_index in session_video_indexes]
    session_video_betas = glmaccess.read_betas(
        sub, ses, glmtype="TYPED_FITHRF_GLMDENOISE_RR", zscore=zscore
    )
    session_video_betas = np.array(session_video_betas)
    print(f"Shape of session video betas: {session_video_betas.shape}")
    mask = glmaccess.read_R2_mask(
        sub, ses=1, threshold=mask_threshold
    )  ######################################## R2 ses should be same for all
    mask_reshape = mask.flatten()
    print(f"Shape of mask: {mask.shape}")
    print(f"Shape of mask reshape: {mask_reshape.shape}")
    # switch session video betas from (x,y,z,t) to (t,x,y,z)
    session_video_betas = np.moveaxis(session_video_betas, 3, 0)

    # get first 720 if the video indexes are more than 720 ############################################################################################################
    session_video_betas = session_video_betas[:720]

    print(f"Shape of session video betas moved: {session_video_betas.shape}")
    # faltten the video betas
    session_video_betas = flatten_video_betas(session_video_betas)
    print(f"Shape of session video betas flattened: {session_video_betas.shape}")

    session_video_betas = session_video_betas[:, mask_reshape]
    print(f"Shape of session video betas masked: {session_video_betas.shape}")
    # if centered:
    #     session_video_betas = session_video_betas - np.mean(session_video_betas, axis=0)
    return session_video_betas

    # get video indexes from session

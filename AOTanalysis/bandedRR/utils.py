import numpy as np
from AOTaccess.glmsingle_access import GLMSingleAccess


def split_single_array(arr, n_splits):
    """
    Splits a numpy array into approximately equal parts.

    Args:
        arr (numpy.ndarray): The input array to be split.
        n_splits (int): The number of parts to split the array into.

    Returns:
        list: A list of numpy arrays, where each array is a part of the original array.

    Notes:
        - If the array cannot be evenly divided, the last part will contain the remaining elements.
        - The function ensures that all elements of the input array are included in the output.
    """
    n_samples = arr.shape[0]
    n_samples_per_split = n_samples // n_splits
    splits = []
    for i in range(n_splits):
        start = i * n_samples_per_split
        end = (i + 1) * n_samples_per_split
        if i == n_splits - 1:
            end = n_samples
        splits.append(arr[start:end])
    return splits


def split_single_list(arr, n_splits):
    """
    Splits a list into approximately equal parts.

    Args:
        arr (list): The input list to be split.
        n_splits (int): The number of parts to split the list into.

    Returns:
        list: A list of sublists, where each sublist is a part of the original list.

    Notes:
        - If the list cannot be evenly divided, the last part will contain the remaining elements.
        - The function ensures that all elements of the input list are included in the output.
    """
    n_samples = len(arr)
    n_samples_per_split = n_samples // n_splits
    splits = []
    for i in range(n_splits):
        start = i * n_samples_per_split
        end = (i + 1) * n_samples_per_split
        if i == n_splits - 1:
            end = n_samples
        splits.append(arr[start:end])
    return splits


def reshape_from_flatten_masked_to_wholebrain(data, sub, ses=1):
    """
    Reshape flattened masked data back to the whole-brain 3D volume.

    Args:
        data (numpy.ndarray): A 1D array containing the flattened and masked 
            brain data. The length of this array corresponds to the number of 
            voxels included in the mask.
        sub (str): The subject identifier. This is used to retrieve the 
            appropriate mask and shape information for the subject.
        ses (int, optional): The session number. Defaults to 1. This is used 
            to retrieve session-specific mask and shape information.

    Returns:
        numpy.ndarray: A 3D array representing the whole-brain data, where the 
            masked data has been placed back into its original spatial 
            locations, and the rest of the brain is filled with zeros.

    Notes:
        - The function relies on the `GLMSingleAccess` class to retrieve the 
          mask and shape information for the specified subject and session.
        - The mask is used to determine which voxels in the whole-brain volume 
          correspond to the input data. The remaining voxels are set to zero.
        - The input data is assumed to be in the same order as the flattened 
          mask.

    Example:
        >>> data = np.array([1.2, 3.4, 5.6])  # Flattened masked data
        >>> sub = "subject01"
        >>> ses = 1
        >>> reshaped_data = reshape_from_flatten_masked_to_wholebrain(data, sub, ses)
        >>> print(reshaped_data.shape)
        (x, y, z)  # Shape of the whole-brain volume for the subject/session
    """
    glmaccess = GLMSingleAccess()
    mask = glmaccess.read_R2_mask(sub, ses=ses)
    shape = glmaccess.read_shape(sub, ses=ses)
    voxel_num = np.prod(shape)
    flatten_mask = mask.flatten()
    data_reshape = np.zeros(voxel_num)
    data_reshape[flatten_mask] = data
    data_reshape = data_reshape.reshape(shape)
    return data_reshape

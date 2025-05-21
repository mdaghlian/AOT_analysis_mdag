# --- Imports ---
import voxelwise_tutorials
from himalaya.ridge import RidgeCV
import numpy as np
from pathlib import Path
from AOTaccess.stimulus_info_access import StimuliInfoAccess
from AOTaccess.glmsingle_access import GLMSingleAccess
from AOTanalysis.bandedRR.construct_features import (
    construct_features_motion_energy_from_subses,
    construct_features_sbert_embeddings_from_subses,
)
from AOTanalysis.bandedRR.construct_target import (
    construct_target_data_from_session_flatten_masked,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from himalaya.backend import set_backend
from himalaya.kernel_ridge import (
    MultipleKernelRidgeCV,
    KernelRidgeCV,
    Kernelizer,
    ColumnKernelizer,
)
from sklearn import set_config
from voxelwise_tutorials.utils import explainable_variance
from AOTanalysis.bandedRR.utils import (
    split_single_array,
    reshape_from_flatten_masked_to_wholebrain,
)
from AOTglmsingle.glmoutput_save_nifti import get_affine_matrix
from himalaya.scoring import r2_score
import os
import joblib
import nibabel as nib
from AOTaccess.expdesign_access import ExpDesignAccess
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import pearsonr
from sklearn.preprocessing import Normalizer


# --- Grouped Function Definitions ---
def data_construct(sub, ses, feature, Ycentered=True, Xcentered=False):
    expdesign_access = ExpDesignAccess()
    video_indexes = expdesign_access.get_session_video_indexes(sub, ses)
    video_betas = construct_target_data_from_session_flatten_masked(
        sub, ses, centered=Ycentered
    )
    if video_betas.shape[0] != len(video_indexes):
        video_betas = video_betas[: len(video_indexes)]
    print(f"Shape of video betas: {video_betas.shape}")
    if feature == "motion":
        X = construct_features_motion_energy_from_subses(sub, ses, centered=Xcentered)
    elif feature == "semantic":
        X = construct_features_sbert_embeddings_from_subses(
            sub, ses, centered=Xcentered
        )
    else:
        raise ValueError("feature must be 'motion' or 'semantic'")
    print(f"Shape of X: {X.shape}")
    y = video_betas
    print(f"Shape of y: {y.shape}")
    Xtrain, Xtest = split_single_array(X, n_splits=2)
    ytrain, ytest = split_single_array(y, n_splits=2)
    print(f"Shape of Xtrain: {Xtrain.shape}")
    print(f"Shape of Xtest: {Xtest.shape}")
    print(f"Shape of ytrain: {ytrain.shape}")
    print(f"Shape of ytest: {ytest.shape}")
    if np.isnan(y).any() or np.isinf(y).any():
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    return Xtrain, ytrain, Xtest, ytest


def mask_construct(R2_file, threshold=0.01):
    R2 = nib.load(R2_file).get_fdata()
    mask = np.zeros_like(R2)
    mask[R2 > threshold] = 1
    return mask


def mask_and_flatten_list_of_data(list_of_data, mask):
    masked_data = [data[mask == 1] for data in list_of_data]
    flattened_masked_data = [data.flatten() for data in masked_data]
    return flattened_masked_data


def scores_for_a_prediction(prediction, betas):
    scores = [pearsonr(prediction, beta)[0] for beta in betas]
    return scores


def get_most_similar_beta_corr(prediction, betas):
    scores = scores_for_a_prediction(prediction, betas)
    most_similar_index = np.argmax(scores)
    return most_similar_index


# --- Main Execution Code wrapped in main() ---
def test_session_inside_general(
    sub,
    train_ses,
    Xcentered,
    Ycentered,
    Xstd,
    feature,
    test_time_normalization,
):
    model_path = f"/tank/shared/2024/visual/AOT/temp/bandedRR_split_single_feature_withSTD_session_testinside/model_sub{sub}_feature_{feature}_trainses_1_Xcentered_{Xcentered}_Ycentered_{Ycentered}_Xstd_{Xstd}_testinside.joblib"
    R2_file = f"/tank/shared/2024/visual/AOT/temp/bandedRR_split_single_feature_withSTD_session_testinside/R2_score_single_sub{sub}_feature_{feature}_trainses_1_Xcentered_{Xcentered}_Ycentered_{Ycentered}_Xstd_{Xstd}_testinside.nii.gz"

    model = joblib.load(model_path)

    Xtrain, ytrain, Xtest, ytest = data_construct(
        sub=sub,
        ses=train_ses,
        feature=feature,
        Ycentered=Ycentered,
        Xcentered=Xcentered,
    )
    print(f"Shape of test_X: {Xtest.shape}")
    print(f"Shape of test_y: {ytest.shape}")

    print(f"Shape of test_video_betas: {ytest.shape}")
    len_test = ytest.shape[0]
    list_of_test_video_betas_glm = [ytest[i] for i in range(len_test)]
    list_of_test_video_betas_glm_wholebrain = [
        reshape_from_flatten_masked_to_wholebrain(
            list_of_test_video_betas_glm[i], sub=sub
        )
        for i in range(len_test)
    ]
    print(
        f"len of list_of_test_video_betas_wholebrain: {len(list_of_test_video_betas_glm_wholebrain)}"
    )
    print(
        f"Shape of list_of_test_video_betas_wholebrain[0]: {list_of_test_video_betas_glm_wholebrain[0].shape}"
    )
    print(f"Shape of test_X: {Xtest.shape}")
    print(f"Shape of test_X: {Xtest.shape}")
    print(
        f"len of list_of_test_video_betas_glm_wholebrain: {len(list_of_test_video_betas_glm_wholebrain)}"
    )
    print(
        f"Shape of list_of_test_video_betas_wholebrain[0]: {list_of_test_video_betas_glm_wholebrain[0].shape}"
    )

    set_backend("torch_cuda", on_error="warn")
    test_X = np.array(Xtest)
    model_predictions = model.predict(test_X)
    print(f"Shape of model_predictions: {model_predictions.shape}")
    model_predictions = np.array(model_predictions)
    print(f"Shape of model_predictions: {model_predictions.shape}")
    len_test = model_predictions.shape[0]
    list_of_model_predictions_wholebrain = [
        reshape_from_flatten_masked_to_wholebrain(model_predictions[i], sub)
        for i in range(len_test)
    ]
    print(
        f"len of list_of_model_predictions_wholebrain: {len(list_of_model_predictions_wholebrain)}"
    )

    prediction_R2_mask = mask_construct(R2_file)
    print(f"Shape of prediction_R2_mask: {prediction_R2_mask.shape}")

    flatten_and_masked_model_predictions = mask_and_flatten_list_of_data(
        list_of_model_predictions_wholebrain, prediction_R2_mask
    )
    flatten_and_masked_test_video_betas_glm = mask_and_flatten_list_of_data(
        list_of_test_video_betas_glm_wholebrain, prediction_R2_mask
    )
    print(
        f"Shape of flatten_and_masked_model_predictions: {flatten_and_masked_model_predictions[0].shape}"
    )
    print(
        f"Shape of flatten_and_masked_test_video_betas_glm: {flatten_and_masked_test_video_betas_glm[0].shape}"
    )

    if test_time_normalization:
        normalizer = Normalizer()
        normalizer.fit(flatten_and_masked_model_predictions)
        flatten_and_masked_model_predictions = normalizer.transform(
            flatten_and_masked_model_predictions
        )
        flatten_and_masked_test_video_betas_glm = normalizer.transform(
            flatten_and_masked_test_video_betas_glm
        )

    plt.plot(
        np.array(flatten_and_masked_model_predictions).mean(1),
        np.array(flatten_and_masked_test_video_betas_glm).mean(1),
        "o",
    )
    plt.savefig(
        f"/tank/shared/2024/visual/AOT/temp/bandedRR_split_single_feature_withSTD_session_testinside/plot_sub{sub}_feature_{feature}_trainses_1_Xcentered_{Xcentered}_Ycentered_{Ycentered}_Xstd_{Xstd}_testinside_average_voxel_corr.png"
    )

    most_similar_betas_corr = [
        get_most_similar_beta_corr(
            flatten_and_masked_model_predictions[i],
            flatten_and_masked_test_video_betas_glm,
        )
        for i in range(len(flatten_and_masked_model_predictions))
    ]
    print(f"Shape of most_similar_betas_corr: {len(most_similar_betas_corr)}")
    print(f"most_similar_betas_corr: {most_similar_betas_corr}")

    similarity_matrix = [
        scores_for_a_prediction(
            flatten_and_masked_model_predictions[i],
            flatten_and_masked_test_video_betas_glm,
        )
        for i in range(len(flatten_and_masked_model_predictions))
    ]
    similarity_matrix = np.array(similarity_matrix)
    print(f"Shape of similarity_matrix: {similarity_matrix.shape}")
    plt.figure(figsize=(10, 10))
    sns.heatmap(similarity_matrix, cmap="coolwarm", center=0)
    plt.show()
    plt.savefig(
        f"/tank/shared/2024/visual/AOT/temp/bandedRR_split_single_feature_withSTD_session_testinside/plot_sub{sub}_feature_{feature}_trainses_1_Xcentered_{Xcentered}_Ycentered_{Ycentered}_Xstd_{Xstd}_testinside_similarity_matrix.png"
    )


if __name__ == "__main__":
    test_session_inside_general(
        sub=3,
        train_ses=1,
        Xcentered=True,
        Ycentered=True,
        Xstd=False,
        feature="semantic",
        test_time_normalization=True,
    )

    test_session_inside_general(
        sub=3,
        train_ses=1,
        Xcentered=True,
        Ycentered=True,
        Xstd=False,
        feature="motion",
        test_time_normalization=True,
    )

    test_session_inside_general(
        sub=2,
        train_ses=1,
        Xcentered=True,
        Ycentered=True,
        Xstd=False,
        feature="semantic",
        test_time_normalization=True,
    )

    test_session_inside_general(
        sub=2,
        train_ses=1,
        Xcentered=True,
        Ycentered=True,
        Xstd=False,
        feature="motion",
        test_time_normalization=True,
    )

    test_session_inside_general(
        sub=1,
        train_ses=1,
        Xcentered=True,
        Ycentered=True,
        Xstd=False,
        feature="semantic",
        test_time_normalization=True,
    )

    test_session_inside_general(
        sub=1,
        train_ses=1,
        Xcentered=True,
        Ycentered=True,
        Xstd=False,
        feature="motion",
        test_time_normalization=True,
    )

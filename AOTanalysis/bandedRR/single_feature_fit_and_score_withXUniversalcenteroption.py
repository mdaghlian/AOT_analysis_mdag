import voxelwise_tutorials
from himalaya.ridge import RidgeCV
import numpy as np
from pathlib import Path
from AOTaccess.stimulus_info_access import StimuliInfoAccess
from AOTaccess.glmsingle_access import GLMSingleAccess
from AOTanalysis.bandedRR.construct_features import (
    construct_features_motion_energy,
    construct_features_sbert_embeddings,
    construct_features_sbert_embeddings_PCA,
)
from AOTanalysis.bandedRR.construct_target import (
    construct_target_data_split_flatten_masked,
)

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from himalaya.backend import set_backend


from himalaya.kernel_ridge import MultipleKernelRidgeCV
from himalaya.kernel_ridge import KernelRidgeCV
from himalaya.kernel_ridge import Kernelizer
from sklearn import set_config
from himalaya.kernel_ridge import ColumnKernelizer
from voxelwise_tutorials.utils import explainable_variance
from AOTanalysis.bandedRR.utils import split_single_array
from AOTglmsingle.glmoutput_save_nifti import get_affine_matrix
from AOTanalysis.bandedRR.utils import reshape_from_flatten_masked_to_wholebrain
from himalaya.scoring import r2_score

import os

import joblib
import nibabel as nib


def data_construct(
    sub,
    n_splits,
    split_index,
    direction,
    feature,
    Ycentered=True,
    Xcentered=False,
):
    video_betas, video_index = construct_target_data_split_flatten_masked(
        sub,
        n_splits,
        split_index,
        centered=Ycentered,
        direction=direction,
        randomize=True,  # randomize the betas and the index in the same way before sample, with a determined seed
    )
    if feature == "motion":
        X = construct_features_motion_energy(video_index, centered=Xcentered)
    elif feature == "semantic":
        X = construct_features_sbert_embeddings(video_index, centered=Xcentered)
    else:
        raise ValueError("feature must be 'motion' or 'semantic'")
    print(f"Shape of X: {X.shape}")

    y = video_betas
    print(f"Shape of y: {y.shape}")

    if np.isnan(y).any() or np.isinf(y).any():
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

    return X, y


def fit_split(
    sub: int,
    n_splits=10,
    split_index=0,
    feature="semantic",
    gpu=True,
    direction="fw",
    save_dir="/tank/shared/2024/visual/AOT/temp/bandedRR_split_single_feature",
):

    def test_and_score(savedir, model):
        Xtest, ytest = data_construct(
            sub=sub,
            n_splits=n_splits,
            split_index=split_index + 1,
            feature=feature,
            direction=direction,
        )
        # use the next split as the test data
        affine = get_affine_matrix(sub=sub)
        general_score = model.score(Xtest, ytest)
        print("shape of general_score", general_score.shape)

        # to numpy
        general_score = np.array(general_score.cpu())
        general_score_reshape = reshape_from_flatten_masked_to_wholebrain(
            general_score, sub
        )
        print("shape of general_score_reshape", general_score_reshape.shape)

        # save the general score
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        generalscoreimg = nib.Nifti1Image(general_score_reshape, affine)
        save_name = f"general_score_sub{sub}_{feature}_train{split_index}_splittotalnum_{n_splits}_test{split_index+1}_{direction}.nii.gz"
        nib.save(generalscoreimg, os.path.join(savedir, save_name))

        test_predict_split = model.predict(Xtest)
        print("shape of test_predict", test_predict_split.shape)
        R2_score = r2_score(ytest, test_predict_split)
        print("shape of R2_score", R2_score.shape)
        R2_score = np.array(R2_score.cpu())
        R2_score_reshape = reshape_from_flatten_masked_to_wholebrain(R2_score, sub)
        print("shape of R2_score_reshape", R2_score_reshape.shape)

        # save the R2 score
        R2scoreimg = nib.Nifti1Image(R2_score_reshape, affine)
        save_name = f"R2_score_single_sub{sub}_{feature}_train{split_index}_splittotalnum_{n_splits}_test{split_index+1}_{direction}.nii.gz"
        nib.save(R2scoreimg, os.path.join(savedir, save_name))

    """apply banded ridge regression to the data belonging to the subject sub"""
    if gpu:
        backend = set_backend("torch_cuda", on_error="warn")
    else:
        backend = set_backend("numpy", on_error="warn")

    X, y = data_construct(
        sub=sub,
        n_splits=n_splits,
        split_index=split_index,
        direction=direction,
        feature=feature,
    )
    # fit the model

    alphas = np.logspace(1, 20, 20)
    n_targets_batch = 200
    n_alphas_batch = 5
    n_targets_batch_refit = 200

    path = Path(save_dir)
    if os.path.exists(path):
        print("Path exists")
    else:
        os.makedirs(path)

    pipeline = make_pipeline(
        StandardScaler(with_mean=True, with_std=False),
        KernelRidgeCV(
            alphas=alphas,
            cv=5,
            solver="eigenvalues",
            solver_params=dict(
                n_targets_batch=n_targets_batch,
                n_alphas_batch=n_alphas_batch,
                n_targets_batch_refit=n_targets_batch_refit,
            ),
        ),
    )

    print("Fitting split ", split_index)
    pipeline.fit(X, y)
    model_path = (
        path
        / f"model_sub{sub}_split{split_index}_splittotalnum_{n_splits}_single_{feature}.joblib"
    )
    joblib.dump(pipeline, model_path)

    savedir = Path(save_dir)

    test_and_score(savedir, model=pipeline)


if __name__ == "__main__":
    sub = 1
    n_splits = 10
    split_index = 0
    gpu = True

    fit_split(
        sub=sub,
        n_splits=n_splits,
        split_index=split_index,
        feature="semantic",
        gpu=gpu,
    )

    fit_split(
        sub=sub,
        n_splits=n_splits,
        split_index=split_index,
        feature="motion",
        gpu=gpu,
    )

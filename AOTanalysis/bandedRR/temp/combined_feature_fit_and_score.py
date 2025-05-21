import voxelwise_tutorials
from himalaya.ridge import RidgeCV
import numpy as np
from pathlib import Path
from AOTaccess.stimulus_info_access import StimuliInfoAccess
from AOTaccess.glmsingle_access import GLMSingleAccess
from AOTanalysis.bandedRR.construct_features import (
    construct_features_motion_energy,
    construct_features_sbert_embeddings,
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
from himalaya.scoring import r2_score_split

import os

import joblib
import nibabel as nib

# the version that feature X is centered by average of sampled training set X it self


def data_construct(sub, n_splits, split_index, direction, feature_names):
    video_betas, video_index = construct_target_data_split_flatten_masked(
        sub,
        n_splits,
        split_index,
        centered=True,
        randomize=True,
        direction=direction,
    )
    y = video_betas
    print(f"Shape of y: {y.shape}")

    if np.isnan(y).any() or np.isinf(y).any():
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

    X_list = []
    for feature_name in feature_names:
        if feature_name == "motion_energy":
            X = construct_features_motion_energy(video_index)
        elif feature_name == "semantic_embeddings":
            X = construct_features_sbert_embeddings(video_index)
        else:
            raise ValueError("feature name do not match")
        print(f"Shape of {feature_name}: {X.shape}")
        X_list.append(X)
    n_features_list = [X.shape[1] for X in X_list]
    X = np.concatenate(X_list, axis=1)
    print(f"Shape of X: {X.shape}")
    return X, y, n_features_list


def fit_split(
    sub: int,
    n_splits=10,
    split_index=0,
    gpu=True,
    Xcentered=True,
    ycentered=True,
    randomize=False,
    feature_names=["motion_energy", "semantic_embeddings"],
    direction="fw",
    save_dir="/tank/shared/2024/visual/AOT/temp/bandedRR_split",
):
    """apply banded ridge regression to the data belonging to the subject sub"""

    def test_and_score(savedir, model):
        Xtest, ytest, n_features_list_test = data_construct(
            sub, n_splits, split_index + 1, direction, feature_names
        )
        # use the next split as the test data
        affine = get_affine_matrix(sub=sub, ses=1)
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
        save_name = f"general_score_sub{sub}_train{split_index}_test{split_index+1}_{direction}.nii.gz"
        nib.save(generalscoreimg, os.path.join(savedir, save_name))

        test_predict_split = model.predict(Xtest, split=True)
        print("shape of test_predict", test_predict_split.shape)

        r2_score_split_output = r2_score_split(ytest, test_predict_split)
        print("shape of r2_score_split", r2_score_split_output.shape)
        r2_score_split_output = np.array(r2_score_split_output.cpu())

        r2_score_split_reshape = np.array(
            [
                reshape_from_flatten_masked_to_wholebrain(r2_score_split_output[i], sub)
                for i in range(r2_score_split_output.shape[0])
            ]
        )

        for i in range(r2_score_split_reshape.shape[0]):
            splitscore = r2_score_split_reshape[i]
            splitscoreimg = nib.Nifti1Image(splitscore, affine)
            save_name = f"splitscore_sub{sub}_train{split_index}_test{split_index+1}_{direction}_{feature_names[i]}.nii.gz"
            nib.save(splitscoreimg, os.path.join(savedir, save_name))

    if gpu:
        backend = set_backend("torch_cuda", on_error="warn")
    else:
        backend = set_backend("numpy", on_error="warn")

    Xtrain, ytrain, n_features_list_train = data_construct(
        sub, n_splits, split_index, direction, feature_names
    )
    # fit the model
    solver = "random_search"
    n_iter = 20

    alphas = np.logspace(1, 20, 20)
    n_targets_batch = 200
    n_alphas_batch = 5
    n_targets_batch_refit = 200
    solver_params = dict(
        n_iter=n_iter,
        alphas=alphas,
        n_targets_batch=n_targets_batch,
        n_alphas_batch=n_alphas_batch,
        n_targets_batch_refit=n_targets_batch_refit,
    )

    savedir = Path(save_dir)

    if os.path.exists(savedir):
        print("Path exists")
    else:
        os.makedirs(savedir)

    mkr_model = MultipleKernelRidgeCV(
        kernels="precomputed",
        solver=solver,
        solver_params=solver_params,
        cv=5,
    )

    if Xcentered:
        preprocess_pipeline = make_pipeline(
            StandardScaler(with_mean=True, with_std=False),
            Kernelizer(kernel="linear"),
        )
    else:
        preprocess_pipeline = make_pipeline(
            Kernelizer(kernel="linear"),
        )

    start_and_end = np.concatenate([[0], np.cumsum(n_features_list_train)])
    slices = [
        slice(start, end) for start, end in zip(start_and_end[:-1], start_and_end[1:])
    ]

    kernelizers_tuples = [
        (name, preprocess_pipeline, slice_)
        for name, slice_ in zip(feature_names, slices)
    ]
    column_kernelizer = ColumnKernelizer(kernelizers_tuples)

    pipeline = make_pipeline(
        column_kernelizer,
        mkr_model,
    )

    print("Fitting split ", split_index)
    pipeline.fit(Xtrain, ytrain)
    model_path = savedir / f"model_sub{sub}_split{split_index}_{direction}.joblib"
    joblib.dump(pipeline, model_path)

    # test the model
    test_and_score(savedir, model=pipeline)


if __name__ == "__main__":
    sub = 1
    n_splits = 10
    for split_index in range(n_splits):
        for direction in ["fw", "rv"]:
            fit_split(
                sub,
                n_splits,
                split_index,
                gpu=True,
                Xcentered=True,
                ycentered=True,
                randomize=True,
                feature_names=["motion_energy", "semantic_embeddings"],
                direction=direction,
                save_dir="/tank/shared/2024/visual/AOT/temp/bandedRR_split_motion_sbert768_test",
            )
        break

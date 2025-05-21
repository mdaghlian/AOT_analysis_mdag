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

from himalaya.kernel_ridge import KernelRidgeCV
from himalaya.kernel_ridge import Kernelizer
from sklearn import set_config
from himalaya.kernel_ridge import ColumnKernelizer
from voxelwise_tutorials.utils import explainable_variance
from AOTanalysis.bandedRR.utils import split_single_array

import os

import joblib


def fit_split(
    sub: int,
    n_splits=10,
    split_index=0,
    feature="semantic",
    gpu=True,
    Xcentered=True,
    ycentered=True,
):
    """apply banded ridge regression to the data belonging to the subject sub"""
    if gpu:
        backend = set_backend("torch_cuda", on_error="warn")
    else:
        backend = set_backend("numpy", on_error="warn")
    video_betas, video_index = construct_target_data_split_flatten_masked(
        sub, n_splits, split_index, centered=ycentered
    )

    if feature == "motion":
        X = construct_features_motion_energy(video_index)
    elif feature == "semantic":
        X = construct_features_sbert_embeddings(video_index)
    else:
        raise ValueError("feature must be 'motion' or 'semantic'")
    print(f"Shape of X: {X.shape}")

    y = video_betas
    print(f"Shape of y: {y.shape}")

    if np.isnan(y).any() or np.isinf(y).any():
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

    # fit the model

    alphas = np.logspace(1, 20, 20)
    n_targets_batch = 200
    n_alphas_batch = 5
    n_targets_batch_refit = 200

    path = Path("/tank/shared/2024/visual/AOT/temp/bandedRR_split")
    if os.path.exists(path):
        print("Path exists")
    else:
        os.makedirs(path)

    if Xcentered:
        pipeline = make_pipeline(
            StandardScaler(with_mean=True, with_std=False),
            KernelRidgeCV(
                alphas=alphas,
                cv=5,
                solver="svd",
                solver_params=dict(
                    n_targets_batch=n_targets_batch,
                    n_alphas_batch=n_alphas_batch,
                    n_targets_batch_refit=n_targets_batch_refit,
                ),
            ),
        )
    else:
        pipeline = make_pipeline(
            KernelRidgeCV(
                alphas=alphas,
                cv=5,
                solver="svd",
                solver_params=dict(
                    n_targets_batch=n_targets_batch,
                    n_alphas_batch=n_alphas_batch,
                    n_targets_batch_refit=n_targets_batch_refit,
                ),
            ),
        )

    print("Fitting split ", split_index)
    pipeline.fit(X, y)
    model_path = path / f"model_sub{sub}_split{split_index}_{feature}.joblib"
    joblib.dump(pipeline, model_path)


if __name__ == "__main__":
    sub = 1
    n_splits = 10
    for split_index in range(n_splits):
        fit_split(sub, n_splits, split_index, feature="semantic")
        fit_split(sub, n_splits, split_index, feature="motion")
        break

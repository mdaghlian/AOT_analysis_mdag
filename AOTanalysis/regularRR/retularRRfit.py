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

# import ridge regression from sklearn
from sklearn.linear_model import Ridge
import joblib

import os


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
    video_betas, video_index = construct_target_data_split_flatten_masked(
        sub, n_splits, split_index, centered=ycentered
    )

    if feature == "motion":
        X = construct_features_motion_energy(video_index)
    elif feature == "semantic":
        X = construct_features_sbert_embeddings(video_index)
    else:
        raise ValueError("feature must be 'motion' or 'semantic'")

    if Xcentered:
        X = X - X.mean(axis=0)

    y = video_betas

    if np.isnan(y).any() or np.isinf(y).any():
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

    # create the ridge regression model
    ridge = Ridge(alpha=1.0)

    # fit the model
    ridge.fit(X, y)
    # save the model
    save_dir = (
        f"/tank/shared/2024/visual/AOT/temp/regularRR_split_xc{Xcentered}_yc{ycentered}"
    )
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    joblib.dump(ridge, f"{save_dir}/sub{sub}_split{split_index}_{feature}.joblib")

    score = ridge.score(X, y)
    print(f"shapo of score: {score.shape}")
    return ridge


if __name__ == "__main__":
    sub = 1
    fit_split(sub, feature="semantic", Xcentered=False, ycentered=False)
    fit_split(sub, feature="motion", Xcentered=False, ycentered=False)

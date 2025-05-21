import voxelwise_tutorials
from himalaya.ridge import RidgeCV
import numpy as np
from pathlib import Path
from AOTaccess.stimulus_info_access import StimuliInfoAccess
from AOTaccess.glmsingle_access import GLMSingleAccess
from AOTaccess.expdesign_access import ExpDesignAccess
from AOTanalysis.bandedRR.construct_features import (
    construct_features_motion_energy,
    construct_features_sbert_embeddings,
    construct_features_sbert_embeddings_PCA,
    construct_features_motion_energy_from_subses,
    construct_features_sbert_embeddings_from_subses,
)
from AOTanalysis.bandedRR.construct_target import (
    construct_target_data_from_session_flatten_masked,
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
from AOTglmsingle.glmoutput_save_nifti import get_affine_matrix, get_header
from AOTanalysis.bandedRR.utils import reshape_from_flatten_masked_to_wholebrain
from himalaya.scoring import r2_score

import os

import joblib
import nibabel as nib


def data_construct(
    sub,
    ses_list: list[int],
    feature,
    Yzscore=True,
    Xcentered=True,
):
    expdesign_access = ExpDesignAccess()
    all_video_betas = []
    all_X = []

    for ses_idx, current_ses in enumerate(ses_list):
        video_indexes = expdesign_access.get_session_video_indexes(sub, current_ses)
        video_betas = construct_target_data_from_session_flatten_masked(
            sub, current_ses, zscore=Yzscore
        )
        if video_betas.shape[0] != len(video_indexes):
            # cut
            video_betas = video_betas[: len(video_indexes)]
        print(f"Shape of video betas for session {current_ses}: {video_betas.shape}")
        all_video_betas.append(video_betas)

        if feature == "motion":
            X_ses = construct_features_motion_energy_from_subses(sub, current_ses, centered=Xcentered, highest_freq=32)
        elif feature == "semantic":
            X_ses = construct_features_sbert_embeddings_from_subses(
                sub, current_ses, centered=Xcentered
            )
        else:
            raise ValueError("feature must be 'motion' or 'semantic'")
        print(f"Shape of X for session {current_ses}: {X_ses.shape}")
        all_X.append(X_ses)

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_video_betas, axis=0)
    
    print(f"Shape of concatenated X: {X.shape}")
    print(f"Shape of concatenated y: {y.shape}")

    if np.isnan(y).any() or np.isinf(y).any():
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

    return X, y


def fit_split(
    sub: int,
    train_ses_list: list[int],
    test_ses: int,
    feature="semantic",
    gpu=True,
    save_dir="/tank/shared/2024/visual/AOT/temp/bandedRR_split_single_feature_withSTD_multiple_session_based",
    Xcentered=True,
    Xstd=True,
    Yzscore=True,
):
    glmsingle_aceess = GLMSingleAccess()
    first_train_ses = train_ses_list[0]
    train_ses_shape = glmsingle_aceess.read_shape(sub, first_train_ses)
    print(f"Shape of first train session ({first_train_ses}): {train_ses_shape}")
    test_ses_shape = glmsingle_aceess.read_shape(sub, test_ses)
    print(f"Shape of test session ({test_ses}): {test_ses_shape}")
    if train_ses_shape != test_ses_shape:
        raise ValueError("The shape of the first train session and test session are different")

    def test_and_score(savedir, model):
        Xtest, ytest = data_construct(
            sub=sub,
            ses_list=[test_ses],
            feature=feature,
            Yzscore=Yzscore,
            Xcentered=Xcentered,
        )
        affine = get_affine_matrix(sub=sub)
        header = get_header(sub=sub)
        general_score = model.score(Xtest, ytest)
        print("shape of general_score", general_score.shape)

        if gpu:
            general_score = np.array(general_score.cpu())
        else:
            general_score = np.array(general_score)
        general_score_reshape = reshape_from_flatten_masked_to_wholebrain(
            general_score, sub, first_train_ses
        )
        print("shape of general_score_reshape", general_score_reshape.shape)

        test_predict_split = model.predict(Xtest)
        print("shape of test_predict", test_predict_split.shape)
        R2_score = r2_score(ytest, test_predict_split)
        print("shape of R2_score", R2_score.shape)
        if gpu:
            R2_score = np.array(R2_score.cpu())
        else:
            R2_score = np.array(R2_score)
        R2_score_reshape = reshape_from_flatten_masked_to_wholebrain(R2_score, sub)
        print("shape of R2_score_reshape", R2_score_reshape.shape)

        R2scoreimg = nib.Nifti1Image(R2_score_reshape, affine=affine, header=header)
        train_ses_str = "_".join(map(str, train_ses_list))
        save_name = (
            f"R2_score_single_sub{sub}_{feature}_train_{train_ses_str}_test_{test_ses}.nii.gz"
        )
        nib.save(R2scoreimg, os.path.join(savedir, save_name))

    if gpu:
        backend = set_backend("torch_cuda", on_error="warn")
    else:
        backend = set_backend("numpy", on_error="warn")

    X, y = data_construct(
        sub,
        train_ses_list,
        feature,
        Yzscore=Yzscore,
        Xcentered=Xcentered,
    )

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
        StandardScaler(with_mean=True, with_std=Xstd),
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

    print("Fitting sessions ", train_ses_list)
    pipeline.fit(X, y)
    train_ses_str = "_".join(map(str, train_ses_list))
    model_path = path / f"model_sub{sub}_feature_{feature}_trainses_{train_ses_str}.joblib"
    joblib.dump(pipeline, model_path)

    savedir = Path(save_dir)

    test_and_score(savedir, model=pipeline)


if __name__ == "__main__":
    sub = 3
    train_ses_list = [1,2,3]
    test_ses = 4
    gpu = True


    ##
    # fit_split(
    #     sub=sub,
    #     train_ses_list=train_ses_list,
    #     test_ses=test_ses,
    #     feature="semantic",
    #     gpu=gpu,
    #     Xcentered=True,
    #     Yzscore=True,
    # )

    fit_split(
        sub=sub,
        train_ses_list=train_ses_list,
        test_ses=test_ses,
        feature="motion",
        gpu=gpu,
        Xcentered=True,
        Yzscore=True,
        Xstd=True,
    )

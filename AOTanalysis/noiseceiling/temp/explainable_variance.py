from AOTaccess.expdesign_access import ExpDesignAccess
from AOTaccess.glmsingle_access import GLMSingleAccess
from pathlib import Path
import numpy as np
import nibabel as nib
import os
import re
import csv
import multiprocessing as mp
from functools import partial


expdesignaccess = ExpDesignAccess()
glmaccess = GLMSingleAccess()

def explanable_variance(sub,ses):
    # 加载affine矩阵和header
    affine = glmaccess.read_affine(sub)
    header = glmaccess.read_header(sub)
    glm_output_folder = Path(
        f"/tank/shared/2024/visual/AOT/derivatives/glmsingle/mainexp_newpreproc/sub-{sub:03d}/ses-{ses:02d}_T1W_nordicstc_1.7mm/TYPED_FITHRF_GLMDENOISE_RR"
    )

    raw_video_index = expdesignaccess.append_all_trails_without_blanks(
        sub, ses)
    unique_video_index = list(set(raw_video_index))
    print("Unique video indexes: ", unique_video_index)
    print("Length of raw video indexes: ", len(raw_video_index))

    dict_betas = {}

    for video_index in unique_video_index:
        video_index_name = re.sub(".mp4", "", video_index)
        video_number = int(re.search("\d+", video_index_name).group()) 
        video_type = re.search("fw|rv", video_index).group()
        # print("Video number: ", video_number)
        # print("Video type: ", video_type)
        # 读取beta值
        video_beta_pair = glmaccess.read_video_betas(
            sub, video_num=video_number, direction=video_type)
        # print("Video beta pair shape: ", video_beta_pair.shape)
        # like (2, 81, 95, 101) : 2 is the number of betas, 81, 95, 101 is the shape of the image

        # 将beta值添加到字典中
        dict_betas[video_index] = video_beta_pair


    betas = []
    mean_betas = []
    for video_index in dict_betas:
        video_beta_pair = dict_betas[video_index]
        betas.append(video_beta_pair[0])
        betas.append(video_beta_pair[1])
        mean = np.mean(video_beta_pair, axis=0)
        print("Mean shape: ", mean.shape)
        mean_betas.append(mean)
    betas = np.array(betas)
    mean_betas = np.array(mean_betas)
    print("Mean betas shape: ", mean_betas.shape)
    print("Betas shape: ", betas.shape)

    total_variance = np.var(betas, axis=0)
    print("Total variance shape: ", total_variance.shape)
    signal_variance = np.var(mean_betas, axis=0)
    print("Signal variance shape: ", signal_variance.shape)

    # Calculate explained variance
    explained_variance = signal_variance / total_variance
    print("Explained variance shape: ", explained_variance.shape)
    # Save explained variance to NIfTI file
    explained_variance_img = nib.Nifti1Image(
        explained_variance, affine, header)
    save_dir = "/tank/shared/2024/visual/AOT/temp/explainable_variance_test"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    output_path = os.path.join(
        save_dir, f"sub-{sub:03d}_ses-{ses:02d}_explainable_variance.nii.gz")
    explained_variance_img.to_filename(output_path)
    print(f"Explained variance saved to {output_path}")


def process_subject(sub, sessions):
    """处理单个被试的所有会话"""
    print(f"Processing subject {sub} with {len(sessions)} sessions")
    # 为每个会话创建一个进程池
    with mp.Pool(processes=min(mp.cpu_count(), len(sessions))) as pool:
        # 使用部分函数将被试ID固定
        func = partial(explanable_variance, sub)
        # 并行映射会话
        pool.map(func, sessions)
    print(f"Completed all sessions for subject {sub}")


if __name__ == "__main__":
    sub3 = 3
    sub3sessions = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    sub2 = 2
    sub2sessions = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    sub1 = 1
    sub1sessions = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    # 创建一个进程池来并行处理不同被试
    subjects = [(sub1, sub1sessions), (sub2, sub2sessions), (sub3, sub3sessions)]
    
    # 可选：串行处理被试但并行处理每个被试的会话
    for sub, sessions in subjects:
        process_subject(sub, sessions)
    
    # 或者完全并行处理（被试和会话都并行）
    # with mp.Pool(processes=len(subjects)) as pool:
    #     pool.starmap(process_subject, subjects)
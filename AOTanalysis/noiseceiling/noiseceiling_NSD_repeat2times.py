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




#same condition repeat 2 times(fw and rv considered as different condition)

def noiseceiling_from_z_score_betas(sub, ses):
    condition_repeat = 2

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

    dict_betas = {} # video_index: beta pair

    for video_index in unique_video_index:
        video_index_name = re.sub(".mp4", "", video_index)
        video_number = int(re.search("\d+", video_index_name).group())
        video_type = re.search("fw|rv", video_index).group()
        # print("Video number: ", video_number)
        # print("Video type: ", video_type)
        video_beta_pair = glmaccess.read_video_betas(
            sub, video_num=video_number, direction=video_type, zscore=True)  # zscore=True zscore=True zscore=True zscore=True zscore=True zscore=True zscore=True zscore=True zscore=True zscore=True zscore=True zscore=True
        # print("Video beta pair shape: ", video_beta_pair.shape)
        # like (2, 81, 95, 101) : 2 is the number of betas, 81, 95, 101 is the shape of the image

        dict_betas[video_index] = video_beta_pair

    dict_var_betapairs = {} # video_index: var beta pair
    for video_index in dict_betas:
        video_beta_pair = dict_betas[video_index]
        video_var_beta_pair = np.var(video_beta_pair, axis=0,ddof=1)
        dict_var_betapairs[video_index] = video_var_beta_pair

    sum_var_betapairs = np.zeros_like(list(dict_var_betapairs.values())[0])
    for video_index in dict_var_betapairs:
        video_var_beta_pair = dict_var_betapairs[video_index]
        sum_var_betapairs += video_var_beta_pair
    print("Sum var betapairs shape: ", sum_var_betapairs.shape)
    
    mean_var_betapairs = sum_var_betapairs / len(dict_var_betapairs)
    print("Mean var betapairs shape: ", mean_var_betapairs.shape)
    sigma_noise = np.sqrt(mean_var_betapairs)
    print("Sigma noise shape: ", sigma_noise.shape)

    sigma_signal = np.sqrt(np.abs(1-np.square(sigma_noise)))
    print("Sigma signal shape: ", sigma_signal.shape)

    ncsnr = sigma_signal / sigma_noise
    print("NC-SNR shape: ", ncsnr.shape)

    NC = np.square(ncsnr) / (1/condition_repeat + np.square(ncsnr))

    print("NC shape: ", NC.shape)
    

    save_dir = "/tank/shared/2024/visual/AOT/temp/noiseceiling_NSD_test_repeat_2_times_ddof_1"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    NC_img = nib.Nifti1Image(NC, affine=affine, header=header)
    NC_img.to_filename(
        os.path.join(save_dir, f"sub-{sub:03d}_ses-{ses:02d}_NC.nii.gz")
    )
    print(f"NC saved to {save_dir}/sub-{sub:03d}_ses-{ses:02d}_NC.nii.gz")
  
def process_subject(sub, sessions):

    print(f"Processing subject {sub} with {len(sessions)} sessions")
    with mp.Pool(processes=min(mp.cpu_count(), len(sessions))) as pool:
        func = partial(noiseceiling_from_z_score_betas, sub)
        pool.map(func, sessions)
    print(f"Completed all sessions for subject {sub}")


if __name__ == "__main__":
    sub3 = 3
    sub3sessions = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    sub2 = 2
    sub2sessions = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    sub1 = 1
    sub1sessions = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    subjects = [(sub1, sub1sessions), (sub2, sub2sessions), (sub3, sub3sessions)]
    
    for sub, sessions in subjects:
        process_subject(sub, sessions)
    
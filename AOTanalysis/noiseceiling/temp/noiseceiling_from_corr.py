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


def noiseceiling(sub, ses):
    # 加载affine矩阵和header
    affine = glmaccess.read_affine(sub)
    header = glmaccess.read_header(sub)
    glm_output_folder = Path(
        f"/tank/shared/2024/visual/AOT/derivatives/glmsingle/mainexp_newpreproc/sub-{sub:03d}/ses-{ses:02d}_T1W_nordicstc_1.7mm/TYPED_FITHRF_GLMDENOISE_RR"
    )
    
    # 获取所有视频索引并去重
    raw_video_index = expdesignaccess.append_all_trails_without_blanks(sub, ses)
    unique_video_index = list(set(raw_video_index))
    print("Unique video indexes: ", unique_video_index)
    print("Length of raw video indexes: ", len(raw_video_index))

    # 处理每个视频
    dict_betas = {}

    for video_index in unique_video_index:
        # 从"1479_rv.mp4"提取数字和方向
        video_index_name = re.sub(".mp4", "", video_index)
        video_number = int(re.search("\d+", video_index_name).group())  # 转换为整数
        video_type = re.search("fw|rv", video_index).group()
        # print("Video number: ", video_number)   
        # print("Video type: ", video_type)
        # 读取beta值
        video_beta_pair = glmaccess.read_video_betas(sub,video_num=video_number, direction=video_type)
        # print("Video beta pair shape: ", video_beta_pair.shape)
        # like (2, 81, 95, 101) : 2 is the number of betas, 81, 95, 101 is the shape of the image

        # 将beta值添加到字典中
        dict_betas[video_index]=video_beta_pair
    
    #for each voxel,get the value of the beta
    # Get dimensions from the first beta pair shape
    first_video = list(dict_betas.keys())[0]
    first_beta_pair = dict_betas[first_video]
    _, height, width, depth = first_beta_pair.shape
    print("Voxel dimensions: ", height, width, depth)

    # # Initialize arrays to store voxel-wise correlations
    all_correlations = np.zeros((height, width, depth))
    # valid_voxels_mask = np.zeros((height, width, depth), dtype=bool)

    # Process each voxel
    for x in range(height):
        for y in range(width):
            for z in range(depth):
                # Collect all pairs (x, y) for this voxel
                pairs = []
                for video_index in dict_betas:
                    beta_pair = dict_betas[video_index]
                    x_value = beta_pair[0, x, y, z]
                    y_value = beta_pair[1, x, y, z]
                    pairs.append((x_value, y_value))
                
                # Calculate correlation for this voxel if we have valid pairs
                if len(pairs) > 0:
                    pairs = np.array(pairs)
                    # Check if we have non-zero variance in both components
                    if np.var(pairs[:, 0]) > 0 and np.var(pairs[:, 1]) > 0:
                        correlation = np.corrcoef(pairs[:, 0], pairs[:, 1])[0, 1]
                        all_correlations[x, y, z] = correlation
    #                     valid_voxels_mask[x, y, z] = True

    # Save the correlation map as a nifti file
    correlation_map = nib.Nifti1Image(all_correlations, affine, header)
    save_dir = "/tank/shared/2024/visual/AOT/temp/noiseceilingtest"
    output_path = os.path.join(
        save_dir, f"sub-{sub:03d}_ses-{ses:02d}_noise_ceiling_map.nii.gz"
    )
    # Create the directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # Save the nifti file
    nib.save(correlation_map, output_path)
    print(f"Noise ceiling map saved to {output_path}")


def process_subject(sub, sessions):
    """处理单个被试的所有会话"""
    print(f"Processing subject {sub} with {len(sessions)} sessions")
    # 为每个会话创建一个进程池
    with mp.Pool(processes=min(mp.cpu_count(), len(sessions))) as pool:
        # 使用部分函数将被试ID固定
        func = partial(noiseceiling, sub)
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
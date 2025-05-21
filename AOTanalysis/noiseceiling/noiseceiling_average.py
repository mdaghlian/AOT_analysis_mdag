#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script to calculate the average noise ceiling across sessions for each subject.
对每个subject，所有session的noiseceiling数据作平均

Author: Zhang S
Date: May 15, 2025
"""

import os
import re
import glob
import numpy as np
import nibabel as nib
from tqdm import tqdm


def average_noiseceiling_by_subject(input_dir, output_dir=None):
    """
    Calculate the average noise ceiling for each subject across all sessions.
    
    Parameters
    ----------
    input_dir : str
        Directory containing the noise ceiling files.
    output_dir : str, optional
        Directory to save the averaged noise ceiling files. 
        If None, will use the input directory.
    
    Returns
    -------
    None
    """
    if output_dir is None:
        output_dir = input_dir
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all subjects
    all_files = sorted(glob.glob(os.path.join(input_dir, 'sub-*_ses-*_NC.nii.gz')))
    subject_pattern = re.compile(r'sub-(\d+)')
    
    # Extract unique subject IDs
    subjects = set()
    for f in all_files:
        match = subject_pattern.search(os.path.basename(f))
        if match:
            subjects.add(match.group(1))
    
    print(f"Found {len(subjects)} subjects: {sorted(subjects)}")
    
    # Process each subject
    for subject in tqdm(sorted(subjects), desc="Processing subjects"):
        # Find all sessions for this subject
        subject_files = sorted(glob.glob(os.path.join(input_dir, f'sub-{subject}_ses-*_NC.nii.gz')))
        
        if not subject_files:
            print(f"No files found for subject {subject}")
            continue
        
        print(f"Found {len(subject_files)} sessions for subject {subject}: {[os.path.basename(f) for f in subject_files]}")
        
        # Load all sessions' data
        all_data = []
        img_ref = None  # Reference image for header/affine
        
        for f in subject_files:
            img = nib.load(f)
            if img_ref is None:
                img_ref = img
            all_data.append(img.get_fdata())
        
        # Calculate the average
        avg_data = np.mean(all_data, axis=0)
        
        # Save the result
        output_file = os.path.join(output_dir, f'sub-{subject}_NC_average.nii.gz')
        avg_img = nib.Nifti1Image(avg_data, img_ref.affine, img_ref.header)
        nib.save(avg_img, output_file)
        print(f"Saved average noise ceiling for subject {subject} to {output_file}")
    
    print("All subjects processed.")


if __name__ == "__main__":
    # 直接使用写死的路径
    input_dir = "/tank/shared/2024/visual/AOT/temp/noiseceiling_NSD_same_condition_repeated_4_times_ddof_1"
    output_dir = "/tank/shared/2024/visual/AOT/temp/noiseceiling_NSD_same_condition_repeated_4_times_ddof_1"
    
    # 运行处理函数
    average_noiseceiling_by_subject(input_dir, output_dir)
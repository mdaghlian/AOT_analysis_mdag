from pathlib import Path
import numpy as np
import nibabel as nib
import os
import cortex

prf_NC_path = "/tank/shared/2024/visual/AOT/derivatives/prf/sub-003/prep/sub-003_ses-pRF_task-pRF_rec-nordicstc_run-noiseceiling_part-mag_bold_space-epi_1.7mm.nii.gz"
aot_NC_path = "/tank/shared/2024/visual/AOT/temp/noiseceiling_NSD_test/sub-003_ses-01_NC.nii.gz"

# Load the NIfTI files
prf_nii = nib.load(prf_NC_path)
aot_nii = nib.load(aot_NC_path)

# Get the data arrays
prf_data = prf_nii.get_fdata()
aot_data = aot_nii.get_fdata()

# Transpose the data arrays to match the expected shape (106, 95, 84)
# The original shape was (84, 95, 106), so we swap the first and last axes.
prf_data = prf_data.transpose(2, 1, 0)
aot_data = aot_data.transpose(2, 1, 0)

print(f"PRF data shape after transpose: {prf_data.shape}") # Add print statements to confirm shapes
print(f"AOT data shape after transpose: {aot_data.shape}")

# --- IMPORTANT ---
# Replace 'YOUR_TRANSFORM_NAME' with the correct transform name from your pycortex database
# that maps this EPI data to the subject's reference space (e.g., 'epi_to_anat').
# The 'identity' transform assumes the data is already in the reference space,
# which caused the shape mismatch error because the EPI data (84, 95, 106)
# does not match the reference volume shape (e.g., 512, 512, 512).
transform_name = 'AOT1pt7mm'  # <-- CHANGE THIS

# Create Pycortex Volume objects
# Assuming the data is for subject 'sub-003'
# Adjust 'sub-003' if needed based on your pycortex setup
# Make sure to pass the transform_name as the third argument (xfmname)
prf_vol = cortex.Volume(prf_data, 'sub-003', transform_name, cmap='viridis')
aot_vol = cortex.Volume(aot_data, 'sub-003', transform_name, cmap='plasma')

# Create a dictionary of volumes to display
volumes_to_display = {
    'PRF_NoiseCeiling': prf_vol,
    'AOT_NoiseCeiling': aot_vol
}

# Display the volumes using pycortex webshow
# This will open the visualization in your default web browser
cortex.webshow(volumes_to_display,host='localhost', port=8080, open_browser=True)

# Keep the script running to keep the webshow active if needed,
# or remove this if running interactively (e.g., in Jupyter)
# input("Press Enter to exit...")



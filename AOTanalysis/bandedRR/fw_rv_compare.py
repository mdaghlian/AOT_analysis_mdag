import nibabel as nib


fw_score = "/tank/shared/2024/visual/AOT/temp/bandedRR_split_motion_sbert768_Xprecentered/splitscore_sub1_train0_test1_fw_semantic_embeddings.nii.gz"
rv_score = "/tank/shared/2024/visual/AOT/temp/bandedRR_split_motion_sbert768_Xprecentered/splitscore_sub1_train0_test1_rv_semantic_embeddings.nii.gz"
fw_score_img = nib.load(fw_score)
rv_score_img = nib.load(rv_score)
affine = fw_score_img.affine
fw_score_data = fw_score_img.get_fdata()
rv_score_data = rv_score_img.get_fdata()


# minus the two scores
diff_score = fw_score_data - rv_score_data
# abs
diff_score = abs(diff_score)
diff_score_img = nib.Nifti1Image(diff_score, affine)
nib.save(
    diff_score_img,
    "/tank/shared/2024/visual/AOT/temp/bandedRR_split_motion_sbert768_Xprecentered/abs_diff_score_fw_rv_semantic.nii.gz",
)

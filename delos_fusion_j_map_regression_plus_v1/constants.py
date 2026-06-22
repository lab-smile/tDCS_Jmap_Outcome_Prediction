# constants.py
"""
Global constants for the SMILE ACT tDCS J-Map ML pipeline.
All path and filename constants are defined here for easy modification.
"""

import os

# ─────────────────────────────────────────────
# Random seed
# ─────────────────────────────────────────────
BASE_SEED = 42
NUM_REPETITIONS = 20          # seeds BASE_SEED … BASE_SEED + NUM_REPETITIONS - 1

# ─────────────────────────────────────────────
# EHR / demographics
# ─────────────────────────────────────────────
EHR_FILE_PATH = (
    "/home/junfu.cheng/SMILE/EHR_ACT/data_new/ACT_data_for_Ruogu_04AUG23.xlsx"
)
EHR_PASSWORD = "password"

# ─────────────────────────────────────────────
# Atlas
# ─────────────────────────────────────────────
ATLAS_PATH = (
    "/home/junfu.cheng/SMILE/github/j_map_2025_8_10/"
    "ml_clinical_trial/hammers_atlas/"
    "Hammers_mith_atlas_n30r83_SPM5.nii.gz"
)
ATLAS_LABELS_PATH = (
    "/home/junfu.cheng/SMILE/github/j_map_2025_8_10/"
    "ml_clinical_trial/hammers_atlas/"
    "n30r83_id2name_clean.txt"
)

# ─────────────────────────────────────────────
# J-Map NIfTI files  (magnitude / theta / phi)
# ─────────────────────────────────────────────
JMAP_BASE_DIR = (
    "/blue/camctrp/working/junfu.cheng/"
    "roast_output_spm_registration_auto_accelerated/"
)
JMAP_MAGNITUDE_FILENAME  = "wT1_tDCSLAB_Jbrain.nii"
JMAP_THETA_FILENAME      = "wT1_tDCSLAB_ThetaBrain.nii"
JMAP_PHI_FILENAME        = "wT1_tDCSLAB_PhiBrain.nii"

# ─────────────────────────────────────────────
# J-Map vector components  (x / y / z)
# ─────────────────────────────────────────────
JMAP_VECTOR_BASE_DIR = (
    "/orange/ruogu.fang/junfu.cheng/SMILE/j_map/"
    "j_map_direction/read_matlab_from_skylar/"
)
JMAP_JX_FILENAME = "wT1_tDCSLAB_Jbrain_x_fromEmag.nii"
JMAP_JY_FILENAME = "wT1_tDCSLAB_Jbrain_y_fromEmag.nii"
JMAP_JZ_FILENAME = "wT1_tDCSLAB_Jbrain_z_fromEmag.nii"

# ─────────────────────────────────────────────
# Output
# ─────────────────────────────────────────────
OUTPUT_ROOT = "output"

# ─────────────────────────────────────────────
# Subject lists and labels
# ─────────────────────────────────────────────
TRAINING_SUBJECTS = [
    105256, 105601, 105971, 107802, 109021,
    110081, 115991, 116036, 202251, 202808, 203846,
]
TESTING_SUBJECTS = [
    300700, 301112, 301428, 301513, 301538,
    303009, 303293, 303346, 303673,
]
RESPONDER_LABELS = {
    105256: 1, 105601: 1, 105971: 1, 107802: 1, 109021: 1,
    110081: 0, 115991: 0, 116036: 0, 202251: 0, 202808: 1, 203846: 1,
    300700: 1, 301112: 0, 301428: 1, 301513: 0, 301538: 0,
    303009: 1, 303293: 0, 303346: 1, 303673: 0,
}

# ─────────────────────────────────────────────
# Demographics
# ─────────────────────────────────────────────
DEMOGRAPHIC_VARS = [
    "age_v0", "sex", "gi_marriage", "race", "ethnicity",
    "gi_height", "gi_weight", "bmi", "years_of_education",
]
CONTINUOUS_DEMO_VARS = [
    "age_v0", "gi_height", "gi_weight", "bmi", "years_of_education",
]
CATEGORICAL_DEMO_VARS = ["sex", "gi_marriage", "race", "ethnicity"]

# ─────────────────────────────────────────────────────────────────────────────
# Bayesian optimisation configuration
# ─────────────────────────────────────────────────────────────────────────────
BAYES_OPT_N_ITER   = 10        # number of BO evaluations per inner loop
BAYES_OPT_CV_FOLDS = 3         # inner CV folds used during BO

# n_jobs for cross_val_score inside each BO trial.
# 1  = sequential (safe default)
# -1 = use all cores (faster, requires more memory)
BAYES_OPT_N_JOBS = -1

# n_jobs for the outer repetition loop (20 seeds run in parallel).
# 1  = sequential (safe default, works with or without BO)
# -1 = use all cores (only recommended when use_bayes_opt=False;
#      the code will automatically fall back to 1 when BO is active)
# WARNING: each worker holds its own copy of the NIfTI volumes in RAM.
#          Monitor memory before setting this to -1 on large volumes.
PARALLEL_REPETITIONS_N_JOBS = -1

# Search ranges
PARAM_SPACE_CLF = {
#    "mrmr__max_topk":     (5,   20),
#    "mrmr__min_topk":     (2,   10),
    "rbf__gamma":         (1e-3, 1e1),
    "rbf__n_components":  (10,  200),
}

PARAM_SPACE_REG = PARAM_SPACE_CLF.copy()
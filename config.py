"""
Project: direction_ml
Purpose: Train on MNI-registered current-density direction maps (theta/phi) and evaluate on a held-out test set.
Outputs: ROC/PR curves (PNG/SVG/PDF), metrics table (CSV), and a single PDF summary.

This code is intentionally modular for easier debugging.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional

# ---------------------------------------------------------------------
# Paths (edit these for your environment)
# ---------------------------------------------------------------------

# Old workspace that contains subject-level MNI-warped direction maps
OLD_WORKING_DIR = "/orange/ruogu.fang/junfu.cheng/SMILE/j_map/j_map_direction/read_matlab_from_skylar"

# New workspace to write ML results
NEW_WORKING_DIR = "/orange/ruogu.fang/junfu.cheng/SMILE/j_map/j_map_direction/direction_ml_mixed"

# Expected per-subject files in OLD_WORKING_DIR/<subject_id>/
THETA_MNI_NAME = "wT1_tDCSLAB_theta_fromE_maskJbrain.nii"
PHI_MNI_NAME   = "wT1_tDCSLAB_phi_fromE_maskJbrain.nii"

# ---------------------------------------------------------------------
# Cohorts
# ---------------------------------------------------------------------

TRAIN_IDS_RESPONDER = ["105256","105601","105971","107802","109021","202808","203846"]
TRAIN_IDS_NONRESP   = ["110081","115991","116036","202251"]

TEST_IDS_RESPONDER  = ["300700","301428","303009","303346"]
TEST_IDS_NONRESP    = ["301112","301513","301538","303293","303673"]

# ---------------------------------------------------------------------
# Experiment settings
# ---------------------------------------------------------------------

RANDOM_STATE = 42

# Which hyperparameter search strategy to use:
#   "grid" | "random" | "bayes" | "genetic"
HYPEROPT_STRATEGY = "random"

# Repeated 3-fold CV within training set for hyperparameter tuning
N_SPLITS = 3
N_REPEATS = 1

# Bootstrap settings for test-set uncertainty
BOOTSTRAP_N = 20
BOOTSTRAP_SEED = 40

# ---------------------------------------------------------------------
# Plot/report output names
# ---------------------------------------------------------------------
OUTDIR_NAME = "results"

ROC_PNG = "test_roc.png"
ROC_SVG = "test_roc.svg"
ROC_PDF = "test_roc.pdf"

PR_PNG  = "test_pr.png"
PR_SVG  = "test_pr.svg"
PR_PDF  = "test_pr.pdf"

METRICS_CSV = "test_metrics.csv"
SUMMARY_PDF = "summary_report.pdf"

# -------------------------------------------------------------------------
# FEATURE_MODE controls which channels are exposed to the feature pipeline
# and passed into as_jmap_dataframe().
#
# Expected channel order in X_5d:
#   0: theta   (polar angle from E direction)
#   1: phi     (azimuthal angle from E direction)
#   2: jx      (current density x-component, R→L)
#   3: jy      (current density y-component, A→P)
#   4: jz      (current density z-component, I→S)
#   5: jmag    (|J|, magnitude of current density; Jbrain)
#
# Available options:
#
#   "theta"
#       Use theta only
#       → volume shape (X, Y, Z, 1)
#
#   "phi"
#       Use phi only
#       → volume shape (X, Y, Z, 1)
#
#   "concat"
#       Use theta + phi
#       → volume shape (X, Y, Z, 2)
#       (DEFAULT; backward compatible with earlier models)
#
#   "tp_jmag"
#       Use theta + phi + |J| (Jbrain)
#       → volume shape (X, Y, Z, 3)
#       Useful for jointly modeling directional information and
#       current density magnitude.
#
#   "jxyz"
#       Use current density vector components (jx, jy, jz)
#       → volume shape (X, Y, Z, 3)
#
#   "jmag"
#       Use current density magnitude |J| (Jbrain) only
#       → volume shape (X, Y, Z, 1)
#
#   "all"
#       Use all available channels:
#       theta + phi + jx + jy + jz + |J|
#       → volume shape (X, Y, Z, 6)
# -------------------------------------------------------------------------

FEATURE_MODE = "jxyz"


# ---------------------------
# Split mode
# ---------------------------
# "manual"
#     Use TRAIN_IDS_* and TEST_IDS_* exactly as currently (single held-out test set)
#
# "mixed_site"
#     Pooled UF+UA cohort -> BOOTSTRAP_N repetitions of OUTER_N_SPLITS-fold
#     nested stratified CV (no fixed held-out test set).
#
# "cross_site"
#     Hold out one site (e.g., UA) as test, train on the other (fixed held-out test set).
SPLIT_MODE = "cross_site"

# For "cross_site"
CROSS_SITE_TEST_SITE = "UA"  # or "UF"

# Outer CV on the training split (nested CV outer loop)
OUTER_N_SPLITS = 5

# Progress reporting
REPORT_EVERY_SEC = 60

# pooled cohorts for dynamic splitting
ALL_IDS_RESPONDER = TRAIN_IDS_RESPONDER + TEST_IDS_RESPONDER
ALL_IDS_NONRESP   = TRAIN_IDS_NONRESP   + TEST_IDS_NONRESP


@dataclass(frozen=True)
class SubjectSplit:
    train_ids: List[str]
    train_y: List[int]
    test_ids: List[str]
    test_y: List[int]

    @staticmethod
    def from_lists(
        train_resp: List[str],
        train_non: List[str],
        test_resp: List[str],
        test_non: List[str],
    ) -> "SubjectSplit":
        train_ids = train_resp + train_non
        train_y   = [1]*len(train_resp) + [0]*len(train_non)
        test_ids  = test_resp + test_non
        test_y    = [1]*len(test_resp) + [0]*len(test_non)
        return SubjectSplit(train_ids=train_ids, train_y=train_y, test_ids=test_ids, test_y=test_y)

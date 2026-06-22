# data_loader.py
"""
Loads:
  - EHR / demographic data (ActDataImport)
  - NIfTI j-map volumes for each subject
  - Assembles subject-level DataFrames ready for the ML pipeline.
"""

import io
import os
import warnings
from typing import Dict, List, Optional, Tuple

import msoffcrypto
import nibabel as nib
import numpy as np
import pandas as pd

from constants import (
    EHR_FILE_PATH,
    EHR_PASSWORD,
    JMAP_BASE_DIR,
    JMAP_MAGNITUDE_FILENAME,
    JMAP_THETA_FILENAME,
    JMAP_PHI_FILENAME,
    JMAP_VECTOR_BASE_DIR,
    JMAP_JX_FILENAME,
    JMAP_JY_FILENAME,
    JMAP_JZ_FILENAME,
    TRAINING_SUBJECTS,
    TESTING_SUBJECTS,
    RESPONDER_LABELS,
    DEMOGRAPHIC_VARS,
)


# ─────────────────────────────────────────────────────────────────────────────
# EHR reader
# ─────────────────────────────────────────────────────────────────────────────

class ActDataImport:
    """Decrypt and load the ACT Excel workbook."""

    def __init__(
        self,
        input_dataset_file_path: str = EHR_FILE_PATH,
        passw: str = EHR_PASSWORD,
    ):
        self.input_dataset, self.dictionary = self._read_excel(
            input_dataset_file_path, passw
        )

    @staticmethod
    def _decrypt_file(file_path: str, password: Optional[str]) -> io.BytesIO:
        decrypted = io.BytesIO()
        with open(file_path, "rb") as f:
            office_file = msoffcrypto.OfficeFile(f)
            if password is not None:
                office_file.load_key(password=password)
            office_file.decrypt(decrypted)
        decrypted.seek(0)
        return decrypted

    def _read_excel(
        self, file_path: str, password: Optional[str]
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        file = self._decrypt_file(file_path, password)
        df = pd.read_excel(file, sheet_name="data")
        file.seek(0)
        dictionary = pd.read_excel(file, sheet_name="datadictionary")
        return df, dictionary


# ─────────────────────────────────────────────────────────────────────────────
# NIfTI helpers
# ─────────────────────────────────────────────────────────────────────────────

def _safe_load_nii(path: str) -> Optional[np.ndarray]:
    """Return float32 ndarray or None on failure."""
    try:
        img = nib.load(path)
        return img.get_fdata(dtype=np.float32)
    except Exception as exc:
        print(f"[ERROR] Cannot load {path}: {exc}")
        return None


def load_magnitude(subject_id: int) -> Optional[np.ndarray]:
    path = os.path.join(
        JMAP_BASE_DIR, str(subject_id), JMAP_MAGNITUDE_FILENAME
    )
    return _safe_load_nii(path)


def load_theta(subject_id: int) -> Optional[np.ndarray]:
    path = os.path.join(
        JMAP_BASE_DIR, str(subject_id), JMAP_THETA_FILENAME
    )
    return _safe_load_nii(path)


def load_phi(subject_id: int) -> Optional[np.ndarray]:
    path = os.path.join(
        JMAP_BASE_DIR, str(subject_id), JMAP_PHI_FILENAME
    )
    return _safe_load_nii(path)


def load_jx(subject_id: int) -> Optional[np.ndarray]:
    path = os.path.join(
        JMAP_VECTOR_BASE_DIR, str(subject_id), JMAP_JX_FILENAME
    )
    return _safe_load_nii(path)


def load_jy(subject_id: int) -> Optional[np.ndarray]:
    path = os.path.join(
        JMAP_VECTOR_BASE_DIR, str(subject_id), JMAP_JY_FILENAME
    )
    return _safe_load_nii(path)


def load_jz(subject_id: int) -> Optional[np.ndarray]:
    path = os.path.join(
        JMAP_VECTOR_BASE_DIR, str(subject_id), JMAP_JZ_FILENAME
    )
    return _safe_load_nii(path)


# ─────────────────────────────────────────────────────────────────────────────
# Subject-level demographic / STAI builder
# ─────────────────────────────────────────────────────────────────────────────

def build_subject_demographics(ehr_df: pd.DataFrame) -> pd.DataFrame:
    """
    One row per subject with baseline (tp==1) demographics and:
      - responder label
      - stai_state_tp1, stai_state_tp2, stai_state_decrease (regression target)
    """
    all_subjects = TRAINING_SUBJECTS + TESTING_SUBJECTS
    ehr_df = ehr_df.copy()
    ehr_df["subjectid"] = pd.to_numeric(ehr_df["subjectid"], errors="coerce")
    ehr_df["tp"] = pd.to_numeric(ehr_df["tp"], errors="coerce")

    cols_needed = ["subjectid", "tp"] + DEMOGRAPHIC_VARS + ["stai_state_score"]
    available = [c for c in cols_needed if c in ehr_df.columns]
    df_sel = ehr_df.loc[
        ehr_df["subjectid"].isin(all_subjects) & ehr_df["tp"].isin([1, 2]),
        available,
    ].copy()

    rows = []
    for sid, sub in df_sel.groupby("subjectid"):
        sub = sub.sort_values("tp")
        row: Dict = {"subjectid": sid}

        # demographics: prefer tp==1, fall back to tp==2
        for var in DEMOGRAPHIC_VARS:
            if var not in sub.columns:
                row[var] = np.nan
                continue
            v1 = sub.loc[sub["tp"] == 1, var].dropna()
            v2 = sub.loc[sub["tp"] == 2, var].dropna()
            row[var] = v1.iloc[0] if len(v1) else (v2.iloc[0] if len(v2) else np.nan)

        # STAI
        if "stai_state_score" in sub.columns:
            s1 = sub.loc[sub["tp"] == 1, "stai_state_score"].dropna()
            s2 = sub.loc[sub["tp"] == 2, "stai_state_score"].dropna()
            row["stai_state_tp1"] = s1.iloc[0] if len(s1) else np.nan
            row["stai_state_tp2"] = s2.iloc[0] if len(s2) else np.nan
            if not np.isnan(row["stai_state_tp1"]) and not np.isnan(row["stai_state_tp2"]):
                row["stai_state_decrease"] = (
                    row["stai_state_tp1"] - row["stai_state_tp2"]
                )
            else:
                row["stai_state_decrease"] = np.nan
        else:
            row["stai_state_tp1"] = np.nan
            row["stai_state_tp2"] = np.nan
            row["stai_state_decrease"] = np.nan

        row["responder"] = RESPONDER_LABELS.get(sid, np.nan)
        row["dataset"] = (
            "Training" if sid in TRAINING_SUBJECTS
            else ("Testing" if sid in TESTING_SUBJECTS else "Unknown")
        )
        rows.append(row)

    return pd.DataFrame(rows).set_index("subjectid")


# ─────────────────────────────────────────────────────────────────────────────
# Main assembly: build feature DataFrame for a given input configuration
# ─────────────────────────────────────────────────────────────────────────────

def build_jmap_dataframe(
    subject_ids: List[int],
    input_config: str,
) -> pd.DataFrame:
    """
    Build a DataFrame with one row per subject.
    Columns depend on input_config:
      "magnitude"         -> jmap_mag   (3-D array)
      "theta_phi"         -> jmap_theta, jmap_phi
      "magnitude_theta_phi" -> jmap_mag, jmap_theta, jmap_phi
      "jxyz"              -> jmap_jx, jmap_jy, jmap_jz
      "magnitude_demo"    -> jmap_mag   + demographic columns (numeric only)

    Subjects for which any required file is missing are silently dropped
    and reported.
    """
    records = []
    skipped = []

    for sid in subject_ids:
        row: Dict = {}
        ok = True

        # ---- magnitude ----
        if input_config in ("magnitude", "magnitude_theta_phi", "magnitude_demo"):
            vol = load_magnitude(sid)
            if vol is None:
                skipped.append((sid, "magnitude missing"))
                ok = False
            else:
                row["jmap_mag"] = vol

        # ---- theta / phi ----
        if input_config in ("theta_phi", "magnitude_theta_phi"):
            theta = load_theta(sid)
            phi   = load_phi(sid)
            if theta is None or phi is None:
                skipped.append((sid, "theta/phi missing"))
                ok = False
            else:
                row["jmap_theta"] = theta
                row["jmap_phi"]   = phi

        # ---- Jx / Jy / Jz ----
        if input_config == "jxyz":
            jx = load_jx(sid)
            jy = load_jy(sid)
            jz = load_jz(sid)
            if jx is None or jy is None or jz is None:
                skipped.append((sid, "jxyz missing"))
                ok = False
            else:
                row["jmap_jx"] = jx
                row["jmap_jy"] = jy
                row["jmap_jz"] = jz

        if ok:
            row["subjectid"] = sid
            records.append(row)

    if skipped:
        print("[WARN] Subjects skipped due to missing files:")
        for sid, reason in skipped:
            print(f"       subject {sid}: {reason}")

    df = pd.DataFrame(records).set_index("subjectid")
    return df
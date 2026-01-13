"""
Data I/O utilities.

We assume that for each subject:
OLD_WORKING_DIR/<subject_id>/{THETA_MNI_NAME, PHI_MNI_NAME, ...}

These are MNI-warped (prefix 'w') NIfTI volumes, likely 3D.

We load them and stack them into a 4D array (X,Y,Z,C), then stack subjects into:
(N, X, Y, Z, C)

Default channel order when include_j=True:
    ["theta", "phi", "jx", "jy", "jz", "jmag"]
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Sequence

import numpy as np

try:
    import nibabel as nib
except Exception:
    nib = None

from .config import OLD_WORKING_DIR, THETA_MNI_NAME, PHI_MNI_NAME

# Optional new config constants (recommended to add to config.py, but we also provide fallbacks)
try:
    from .config import JX_MNI_NAME, JY_MNI_NAME, JZ_MNI_NAME, JBRAIN_MNI_NAME
except Exception:
    # Fallbacks based on your new filenames shown in the screenshot
    JX_MNI_NAME = "wT1_tDCSLAB_Jbrain_x_fromEmag.nii"
    JY_MNI_NAME = "wT1_tDCSLAB_Jbrain_y_fromEmag.nii"
    JZ_MNI_NAME = "wT1_tDCSLAB_Jbrain_z_fromEmag.nii"
    JBRAIN_MNI_NAME = "wT1_tDCSLAB_Jbrain.nii"


class DataIOError(RuntimeError):
    pass


@dataclass
class LoadedDataset:
    X: np.ndarray  # shape: (n_subjects, X, Y, Z, C)
    y: np.ndarray  # shape: (n_subjects,)
    ids: List[str]
    channel_names: List[str]
    affine: Optional[np.ndarray] = None
    header: Optional[object] = None


def _require_nibabel():
    if nib is None:
        raise DataIOError(
            "nibabel is required to load NIfTI files. "
            "Install with: pip install nibabel"
        )


def subject_paths(subject_id: str, old_working_dir: str = OLD_WORKING_DIR, include_j: bool = True) -> Dict[str, Path]:
    """
    Returns a dict of expected file paths for a subject.
    """
    sdir = Path(old_working_dir) / str(subject_id)
    paths: Dict[str, Path] = {
        "theta": sdir / THETA_MNI_NAME,
        "phi":   sdir / PHI_MNI_NAME,
    }
    if include_j:
        paths.update(
            {
                "jx": sdir / JX_MNI_NAME,
                "jy": sdir / JY_MNI_NAME,
                "jz": sdir / JZ_MNI_NAME,
                "jmag": sdir / JBRAIN_MNI_NAME,  # magnitude of J (Jbrain)
            }
        )
    return paths


def _load_nifti(path: Path):
    _require_nibabel()
    if not path.exists():
        raise DataIOError(f"Missing file: {path}")
    img = nib.load(str(path))
    data = np.asanyarray(img.dataobj)
    return img, data


def load_subject_volumes(
    subject_id: str,
    old_working_dir: str = OLD_WORKING_DIR,
    include_j: bool = True,
    channel_order: Optional[Sequence[str]] = None,
) -> Tuple[Dict[str, np.ndarray], np.ndarray, object, List[str]]:
    """
    Load all requested channels for one subject.

    Args:
        include_j: if True, loads jx/jy/jz/jmag in addition to theta/phi.
        channel_order: optional explicit ordering of channels. If None, uses:
            - include_j=False: ["theta","phi"]
            - include_j=True:  ["theta","phi","jx","jy","jz","jmag"]

    Returns:
        vols_by_name: dict name -> 3D array
        affine: affine of the first loaded image
        header: header of the first loaded image
        ordered_names: channel names in the order they should be stacked
    """
    paths = subject_paths(subject_id, old_working_dir=old_working_dir, include_j=include_j)

    if channel_order is None:
        ordered_names = ["theta", "phi"] + (["jx", "jy", "jz", "jmag"] if include_j else [])
    else:
        ordered_names = list(channel_order)

    # Ensure requested channels exist in paths
    for ch in ordered_names:
        if ch not in paths:
            raise DataIOError(
                f"Channel '{ch}' not available for subject {subject_id}. "
                f"Available: {sorted(paths.keys())}"
            )

    vols_by_name: Dict[str, np.ndarray] = {}
    affine = None
    header = None
    ref_shape = None

    for ch in ordered_names:
        img, data = _load_nifti(paths[ch])
        if affine is None:
            affine = img.affine
            header = img.header
            ref_shape = data.shape

        if data.shape != ref_shape:
            raise DataIOError(
                f"Shape mismatch for subject {subject_id} on channel '{ch}': "
                f"expected {ref_shape}, got {data.shape}"
            )

        vols_by_name[ch] = data

    return vols_by_name, affine, header, ordered_names


def stack_channels(vols_by_name: Dict[str, np.ndarray], ordered_names: Sequence[str]) -> np.ndarray:
    """
    Stack channels into (X,Y,Z,C) in the provided order.
    """
    chans = []
    for name in ordered_names:
        arr = vols_by_name[name].astype(np.float32, copy=False)
        chans.append(arr)
    return np.stack(chans, axis=-1)


def load_dataset(
    ids: List[str],
    y: List[int],
    old_working_dir: str = OLD_WORKING_DIR,
    include_j: bool = True,
    channel_order: Optional[Sequence[str]] = None,
) -> LoadedDataset:
    """
    Loads all subjects into a single array.

    Returns:
        LoadedDataset with X shape (N,X,Y,Z,C)
    """
    _require_nibabel()
    y_arr = np.asarray(y, dtype=int)
    if len(ids) != len(y_arr):
        raise DataIOError("ids and y must have same length")

    vols = []
    affine = None
    header = None
    channel_names: List[str] = []

    for sid in ids:
        vols_by_name, aff, hdr, ordered_names = load_subject_volumes(
            sid,
            old_working_dir=old_working_dir,
            include_j=include_j,
            channel_order=channel_order,
        )
        if affine is None:
            affine = aff
            header = hdr
            channel_names = list(ordered_names)

        vol4d = stack_channels(vols_by_name, ordered_names)  # (X,Y,Z,C)
        vols.append(vol4d)

    X = np.stack(vols, axis=0)  # (N,X,Y,Z,C)
    return LoadedDataset(X=X, y=y_arr, ids=list(ids), channel_names=channel_names, affine=affine, header=header)

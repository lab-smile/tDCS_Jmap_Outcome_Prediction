from __future__ import annotations
from typing import Literal, Sequence
import numpy as np
import pandas as pd


# Channel indices assume DataIO channel order:
# ["theta", "phi", "jx", "jy", "jz", "jmag"]
ChannelMode = Literal[
    "concat",      # theta+phi
    "theta",
    "phi",
    "jxyz",        # jx+jy+jz
    "jmag",        # |J|
    "tp_jmag",     # theta+phi+|J|
    "all",         # theta+phi+jx+jy+jz+jmag
]


def as_jmap_dataframe(
    X_5d: np.ndarray,
    ids: Sequence[str],
    mode: ChannelMode = "concat",
    colname: str = "jmap_tp1",
) -> pd.DataFrame:
    """
    Build the DataFrame expected by JmapACTPreprocessor.

    Parameters
    ----------
    X_5d : np.ndarray
        Shape (N, X, Y, Z, C).
        Expected channel order (recommended):
            0: theta
            1: phi
            2: jx
            3: jy
            4: jz
            5: jmag  (magnitude |J|, i.e., Jbrain)
    ids : Sequence[str]
        Subject IDs, length N.
    mode : {"concat","theta","phi","jxyz","jmag","tp_jmag","all"}
        - "concat":  store (X,Y,Z,2) per subject (theta+phi)
        - "theta":   store (X,Y,Z,1) per subject (theta only)
        - "phi":     store (X,Y,Z,1) per subject (phi only)
        - "jxyz":    store (X,Y,Z,3) per subject (jx+jy+jz)
        - "jmag":    store (X,Y,Z,1) per subject (|J|)
        - "tp_jmag": store (X,Y,Z,3) per subject (theta+phi+|J|)
        - "all":     store (X,Y,Z,6) per subject (theta+phi+jx+jy+jz+|J|)
    colname : str
        Name of the volume column (default "jmap_tp1").

    Returns
    -------
    pd.DataFrame
        Columns: ["subject_id", colname]
        Each cell in colname contains a numpy array volume.
    """
    X_5d = np.asarray(X_5d)

    if X_5d.ndim != 5:
        raise ValueError(f"Expected X_5d with 5 dims (N,X,Y,Z,C), got shape {X_5d.shape}")

    n = X_5d.shape[0]
    if len(ids) != n:
        raise ValueError(f"ids length ({len(ids)}) must match X_5d first dim ({n})")

    c = X_5d.shape[-1]

    if mode == "concat":
        if c < 2:
            raise ValueError(f"mode='concat' requires at least 2 channels (theta,phi), got C={c}")
        vols = [X_5d[i, ..., 0:2] for i in range(n)]  # (X,Y,Z,2)

    elif mode == "theta":
        if c < 1:
            raise ValueError("mode='theta' requires at least 1 channel (theta)")
        vols = [X_5d[i, ..., 0:1] for i in range(n)]  # (X,Y,Z,1)

    elif mode == "phi":
        if c < 2:
            raise ValueError("mode='phi' requires at least 2 channels (theta,phi) with phi at index 1")
        vols = [X_5d[i, ..., 1:2] for i in range(n)]  # (X,Y,Z,1)

    elif mode == "jxyz":
        if c < 5:
            raise ValueError("mode='jxyz' requires channels jx,jy,jz at indices 2,3,4 (C>=5)")
        vols = [X_5d[i, ..., 2:5] for i in range(n)]  # (X,Y,Z,3)

    elif mode == "jmag":
        if c < 6:
            raise ValueError("mode='jmag' requires |J| (jmag) at index 5 (C>=6)")
        vols = [X_5d[i, ..., 5:6] for i in range(n)]  # (X,Y,Z,1)

    elif mode == "tp_jmag":
        # theta (0) + phi (1) + |J| (5)
        if c < 6:
            raise ValueError("mode='tp_jmag' requires theta,phi and |J| at indices 0,1,5 (C>=6)")
        vols = [X_5d[i, ..., [0, 1, 5]] for i in range(n)]  # (X,Y,Z,3)

    elif mode == "all":
        if c < 6:
            raise ValueError("mode='all' requires 6 channels [theta,phi,jx,jy,jz,jmag] (C>=6)")
        vols = [X_5d[i, ..., 0:6] for i in range(n)]  # (X,Y,Z,6)

    else:
        raise ValueError(f"Unknown mode: {mode}")

    return pd.DataFrame(
        {
            "subject_id": list(ids),
            colname: vols,
        }
    )

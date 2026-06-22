# preprocessors.py
"""
JmapACTPreprocessor  – converts raw 3-D/4-D NIfTI arrays to tabular features.
DFStandardScaler     – StandardScaler that preserves DataFrame structure.
SafeSMOTE            – SMOTE with guard-rails for small / imbalanced folds.
PCAWithNames         – PCA wrapper that returns named-column DataFrames.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import nibabel as nib
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
from typing import List, Optional


# ─────────────────────────────────────────────────────────────────────────────
# JmapACTPreprocessor
# ─────────────────────────────────────────────────────────────────────────────

class JmapACTPreprocessor(BaseEstimator, TransformerMixin):
    """
    Preprocesses 3-D/4-D brain volumes stored in a DataFrame into a
    tabular feature matrix.

    Parameters
    ----------
    jmap_features : list of str
        Column names in the input DataFrame that contain NumPy volume arrays.
    strategy : {"stats", "flatten", "pca"}
        How to reduce each volume to a feature vector.
    n_components : int or None
        Number of PCA components (only used when strategy=="pca").
    keep_channel_axis : bool
        If True, a 4-D volume (X,Y,Z,C) is treated as C-channel voxels.
    random_state : int or None
    atlas_path : str or None
        Path to a NIfTI atlas (Hammers or similar) for region mapping.
    atlas_labels_path : str or None
        Path to the atlas label text file.
    scale_volume : bool
        Apply StandardScaler to the final feature matrix when True.
    verbose : bool
    """

    def __init__(
        self,
        jmap_features: List[str],
        strategy: str = "stats",
        n_components: Optional[int] = 50,
        keep_channel_axis: bool = True,
        random_state: Optional[int] = 0,
        atlas_path: Optional[str] = None,
        atlas_labels_path: Optional[str] = None,
        scale_volume: bool = True,
        verbose: bool = False,
    ):
        self.jmap_features      = jmap_features
        self.strategy           = strategy
        self.n_components       = n_components
        self.keep_channel_axis  = keep_channel_axis
        self.random_state       = random_state
        self.scale_volume       = scale_volume
        self.verbose            = verbose
        self.atlas_path         = atlas_path
        self.atlas_labels_path  = atlas_labels_path

        self._scaler: Optional[StandardScaler] = None
        self._pca:    Optional[PCA]            = None
        self.feature_names_: Optional[List[str]] = None
        self.key_predictors_: Optional[List[str]] = None

        self.atlas_img  = None
        self.atlas_data = None
        self.id2name: Optional[dict] = None

        if atlas_path and atlas_labels_path:
            self._load_atlas()

    # ── Atlas ─────────────────────────────────────────────────────────────
    def _load_atlas(self):
        self.atlas_img  = nib.load(self.atlas_path)
        self.atlas_data = self.atlas_img.get_fdata().astype(int)
        id2name = {}
        with open(self.atlas_labels_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                id2name[int(parts[0])] = " ".join(parts[1:])
        self.id2name = id2name

    def voxel_to_region(self, i: int, j: int, k: int) -> str:
        lab = int(self.atlas_data[i, j, k])
        return self.id2name.get(lab, "Background/Unlabeled")

    def voxel_to_mni(self, i: int, j: int, k: int):
        return tuple(self.atlas_img.affine.dot([i, j, k, 1])[:3])

    def mni_to_region(self, x: float, y: float, z: float) -> str:
        ijk = np.linalg.inv(self.atlas_img.affine).dot([x, y, z, 1])[:3]
        i, j, k = np.round(ijk).astype(int)
        if (
            0 <= i < self.atlas_data.shape[0]
            and 0 <= j < self.atlas_data.shape[1]
            and 0 <= k < self.atlas_data.shape[2]
        ):
            return self.voxel_to_region(i, j, k)
        return "Out of bounds"

    def feature_to_voxel(self, feature_name: str) -> dict:
        if self.strategy != "flatten":
            raise ValueError("Voxel mapping only available for 'flatten'.")
        idx = int(feature_name.split("_")[-1])
        xyz_shape = (
            self._example_shape[:-1]
            if self.keep_channel_axis and len(self._example_shape) == 4
            else self._example_shape
        )
        channels = (
            self._example_shape[-1]
            if self.keep_channel_axis and len(self._example_shape) == 4
            else 1
        )
        voxel_linear = idx // channels
        ch = idx % channels if channels > 1 else None
        i, j, k = np.unravel_index(voxel_linear, xyz_shape)
        region    = self.voxel_to_region(i, j, k) if self.atlas_data is not None else None
        mni_coords = self.voxel_to_mni(i, j, k)   if self.atlas_img  is not None else None
        return {"voxel": (i, j, k), "channel": ch, "mni": mni_coords, "region": region}

    # ── Core helpers ──────────────────────────────────────────────────────
    @staticmethod
    def _to_2d(arr: np.ndarray, keep_channel_axis: bool) -> np.ndarray:
        arr = np.asarray(arr)
        if arr.ndim == 3:
            return arr.reshape(-1, 1)
        if arr.ndim == 4:
            if keep_channel_axis:
                return arr.reshape(np.prod(arr.shape[:-1], dtype=int), arr.shape[-1])
            return arr.reshape(-1, 1)
        return arr.reshape(-1, 1)

    @staticmethod
    def _per_sample_zscore(M: np.ndarray) -> np.ndarray:
        mean = np.nanmean(M, axis=0, keepdims=True)
        std  = np.nanstd(M,  axis=0, keepdims=True)
        std  = np.where(std == 0, 1.0, std)
        return (M - mean) / std

    def _extract_stats(self, M: np.ndarray) -> np.ndarray:
        feats = []
        for ch in range(M.shape[1]):
            v = M[:, ch]
            v = v[np.isfinite(v)]
            if v.size == 0:
                feats.extend([np.nan] * 10)
                continue
            feats.extend([
                np.mean(v), np.std(v), np.min(v), np.max(v), np.median(v),
                np.percentile(v, 1),  np.percentile(v, 5),
                np.percentile(v, 95), np.percentile(v, 99),
                np.count_nonzero(v),
            ])
        return np.array(feats, dtype=float)

    def _prepare_feature_matrix(self, X: pd.DataFrame) -> np.ndarray:
        matrices = []
        for _, row in X.iterrows():
            row_blocks = []
            for col in self.jmap_features:
                M = self._to_2d(row[col], self.keep_channel_axis)
                if not hasattr(self, "_example_shape"):
                    self._example_shape = row[col].shape
                M = self._per_sample_zscore(M)
                if self.strategy in ("flatten", "pca"):
                    row_blocks.append(M.reshape(1, -1))
                elif self.strategy == "stats":
                    row_blocks.append(self._extract_stats(M)[None, :])
                else:
                    raise ValueError(f"Unknown strategy: {self.strategy}")
            matrices.append(np.concatenate(row_blocks, axis=1))
        return np.concatenate(matrices, axis=0)

    @staticmethod
    def _fill_invalid_with_row_mean(matrices: np.ndarray) -> np.ndarray:
        for i in range(matrices.shape[0]):
            row  = matrices[i, :]
            mask = ~np.isfinite(row)
            if np.any(mask):
                valid_mean = np.nanmean(row)
                if np.isnan(valid_mean):
                    valid_mean = 0.0
                row[mask] = valid_mean
        return matrices

    # ── Fit / Transform ───────────────────────────────────────────────────
    def fit(self, X: pd.DataFrame, y=None):
        matrices = self._prepare_feature_matrix(X)
        matrices = self._fill_invalid_with_row_mean(matrices)

        if self.strategy == "pca":
            if self.n_components is None:
                raise ValueError("n_components must be set for 'pca'.")
            self._scaler = StandardScaler()
            ms = self._scaler.fit_transform(matrices)
            if not np.isfinite(ms).all():
                raise ValueError("Non-finite values remain before PCA.")
            self._pca = PCA(
                n_components=self.n_components, random_state=self.random_state
            )
            self._pca.fit(ms)
            self.feature_names_ = [
                f"jmap_pca_{i:03d}" for i in range(self.n_components)
            ]
        else:
            if self.scale_volume:
                self._scaler = StandardScaler()
                self._scaler.fit_transform(matrices)   # fit only
                self._scaler.fit(matrices)
            if self.strategy == "flatten":
                self.feature_names_ = [
                    f"jmap_flat_{i:07d}" for i in range(matrices.shape[1])
                ]
            else:  # stats
                first = self._to_2d(
                    X.iloc[0][self.jmap_features[0]], self.keep_channel_axis
                )
                channels = first.shape[1]
                stat_names = (
                    "mean", "std", "min", "max", "median",
                    "p01", "p05", "p95", "p99", "nnz",
                )
                names = []
                for jf in self.jmap_features:
                    for ch in range(channels):
                        names += [f"{jf}_ch{ch}_{s}" for s in stat_names]
                self.feature_names_ = names
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        matrices = self._prepare_feature_matrix(X)
        matrices = self._fill_invalid_with_row_mean(matrices)

        if self.strategy == "pca" and self._pca is not None:
            ms       = self._scaler.transform(matrices)
            matrices = self._pca.transform(ms)
        elif self.scale_volume and self._scaler is not None:
            matrices = self._scaler.transform(matrices)

        return pd.DataFrame(
            matrices,
            index=X.index,
            columns=self.get_feature_names_out(),
        )

    def get_feature_names_out(self, input_features=None):
        return (
            np.array(self.feature_names_)
            if self.feature_names_ is not None
            else None
        )

    def set_key_predictors(self, predictor_names: List[str]):
        self.key_predictors_ = predictor_names


# ─────────────────────────────────────────────────────────────────────────────
# DFStandardScaler
# ─────────────────────────────────────────────────────────────────────────────

class DFStandardScaler(BaseEstimator, TransformerMixin):
    """StandardScaler that always returns a DataFrame with the same columns."""

    def __init__(self, with_mean: bool = True, with_std: bool = True):
        self.with_mean = with_mean
        self.with_std  = with_std

    def fit(self, X, y=None):
        Xv = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
        self._scaler = StandardScaler(
            with_mean=self.with_mean, with_std=self.with_std
        )
        self._scaler.fit(Xv)
        self.columns_ = (
            list(X.columns) if isinstance(X, pd.DataFrame)
            else [f"f{i}" for i in range(Xv.shape[1])]
        )
        return self

    def transform(self, X):
        Xv = (
            X[self.columns_].values
            if isinstance(X, pd.DataFrame)
            else np.asarray(X)
        )
        idx = getattr(X, "index", None)
        return pd.DataFrame(
            self._scaler.transform(Xv), index=idx, columns=self.columns_
        )

    def get_feature_names_out(self, input_features=None):
        return np.array(self.columns_)


# ─────────────────────────────────────────────────────────────────────────────
# SafeSMOTE
# ─────────────────────────────────────────────────────────────────────────────

class SafeSMOTE(SMOTE):
    """SMOTE with automatic k_neighbors adjustment for very small minority classes."""

    def fit_resample(self, X, y):
        le    = LabelEncoder()
        y_enc = le.fit_transform(y)
        counts = np.bincount(y_enc)

        if (counts == 0).any() or counts.min() == 0:
            return X, y

        if counts.min() >= counts.max():
            return X, y

        if hasattr(self, "k_neighbors"):
            k = getattr(self.k_neighbors, "n_neighbors", self.k_neighbors)
            if counts.min() - 1 < k:
                self.k_neighbors = max(1, counts.min() - 1)

        return super().fit_resample(X, y)


# ─────────────────────────────────────────────────────────────────────────────
# PCAWithNames
# ─────────────────────────────────────────────────────────────────────────────

class PCAWithNames(BaseEstimator, TransformerMixin):
    """
    Thin sklearn-compliant PCA wrapper that:
      1. Returns a named-column DataFrame from transform().
      2. Exposes get_feature_names_out() for downstream selectors.
    """

    def __init__(self, n_components=None, random_state=None):
        self.n_components = n_components
        self.random_state = random_state

    def fit(self, X, y=None):
        self.pca_ = PCA(
            n_components=self.n_components, random_state=self.random_state
        )
        self.pca_.fit(X)
        n_out = self.pca_.n_components_
        self.feature_names_out_ = [f"f{i}" for i in range(n_out)]
        return self

    def transform(self, X):
        Z   = self.pca_.transform(X)
        idx = getattr(X, "index", None)
        return pd.DataFrame(Z, index=idx, columns=self.feature_names_out_)

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X, y).transform(X)

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_out_)
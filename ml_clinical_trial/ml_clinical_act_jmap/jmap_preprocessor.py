import numpy as np
import pandas as pd
import nibabel as nib
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from typing import List, Optional


class JmapACTPreprocessor(BaseEstimator, TransformerMixin):
    """
    Preprocesses 3D/4D brain volumes into tabular features for ML.

    Features:
    ---------
    - Supports 'stats', 'flatten', and 'pca' strategies
    - Optionally applies StandardScaler to all strategies (default True)
    - Maps flatten features back to voxel coordinates and brain regions
    - Assumes X is already in MNI space (atlas grid matches X grid)
    """

    def __init__(
        self,
        jmap_features: List[str],
        strategy: str = "stats",       # "stats", "flatten", or "pca"
        n_components: Optional[int] = 50,  # Only used for "pca"
        keep_channel_axis: bool = True,
        random_state: Optional[int] = 0,
        atlas_path: Optional[str] = None,
        atlas_labels_path: Optional[str] = None,
        scale_volume: bool = True,      # Always scale final features if True
        verbose: bool = False
    ):
        self.jmap_features = jmap_features
        self.strategy = strategy
        self.n_components = n_components
        self.keep_channel_axis = keep_channel_axis
        self.random_state = random_state
        self.scale_volume = scale_volume
        self.verbose = verbose

        # Standardization + PCA
        self._scaler: Optional[StandardScaler] = None
        self._pca: Optional[PCA] = None

        # Feature names and key predictors
        self.feature_names_: Optional[List[str]] = None
        self.key_predictors_: Optional[List[str]] = None

        # Atlas loading
        self.atlas_path = atlas_path
        self.atlas_labels_path = atlas_labels_path
        self.atlas_img = None
        self.atlas_data = None
        self.id2name = None

        if atlas_path and atlas_labels_path:
            self._load_atlas()

    # ------------------------
    # Atlas loading and mapping
    # ------------------------
    def _load_atlas(self):
        """Load atlas and label mapping (no resampling; assumes MNI space)."""
        #print(self.atlas_path)
        atlas_path = "/home/junfu.cheng/SMILE/github/j_map_2025_8_10/ml_clinical_trial/hammers_atlas/Hammers_mith_atlas_n30r83_SPM5.nii.gz"
        self.atlas_img = nib.load(atlas_path)
        self.atlas_data = self.atlas_img.get_fdata().astype(int)

        id2name = {}
        #print(self.atlas_labels_path)
        atlas_labels_path = "/home/junfu.cheng/SMILE/github/j_map_2025_8_10/ml_clinical_trial/hammers_atlas/n30r83_id2name_clean.txt"
        with open(atlas_labels_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                lab_id = int(parts[0])
                name = " ".join(parts[1:])
                id2name[lab_id] = name
        self.id2name = id2name

    def voxel_to_region(self, i: int, j: int, k: int) -> str:
        """Return region name for voxel indices in atlas grid."""
        lab = int(self.atlas_data[i, j, k])
        return self.id2name.get(lab, "Background/Unlabeled")

    def voxel_to_mni(self, i: int, j: int, k: int):
        """Convert voxel indices to MNI coordinates (mm)."""
        return tuple(self.atlas_img.affine.dot([i, j, k, 1])[:3])

    def mni_to_region(self, x: float, y: float, z: float) -> str:
        """Return region for given MNI coords (mm)."""
        ijk = np.linalg.inv(self.atlas_img.affine).dot([x, y, z, 1])[:3]
        i, j, k = np.round(ijk).astype(int)
        if (0 <= i < self.atlas_data.shape[0] and
            0 <= j < self.atlas_data.shape[1] and
            0 <= k < self.atlas_data.shape[2]):
            return self.voxel_to_region(i, j, k)
        return "Out of bounds"

    def feature_to_voxel(self, feature_name: str):
        """
        Given a feature name (from get_feature_names_out),
        return voxel coords, MNI coords, and brain region.
        Only works for 'flatten' strategy (direct voxel mapping).
        """
        if self.strategy != "flatten":
            raise ValueError("Voxel mapping only available for 'flatten' strategy.")

        idx = int(feature_name.split("_")[-1])

        xyz_shape = self._example_shape[:-1] if self.keep_channel_axis and len(self._example_shape) == 4 else self._example_shape
        channels = self._example_shape[-1] if self.keep_channel_axis and len(self._example_shape) == 4 else 1

        voxel_linear = idx // channels
        ch = idx % channels if channels > 1 else None
        i, j, k = np.unravel_index(voxel_linear, xyz_shape)

        region = self.voxel_to_region(i, j, k) if self.atlas_data is not None else None
        mni_coords = self.voxel_to_mni(i, j, k) if self.atlas_img is not None else None

        return {"voxel": (i, j, k), "channel": ch, "mni": mni_coords, "region": region}

    # ------------------------
    # Core helpers
    # ------------------------
    @staticmethod
    def _to_2d(sample_arr: np.ndarray, keep_channel_axis: bool) -> np.ndarray:
        arr = np.asarray(sample_arr)
        if arr.ndim == 3:
            return arr.reshape(-1, 1)
        if arr.ndim == 4:
            if keep_channel_axis:
                voxels = np.prod(arr.shape[:-1], dtype=int)
                channels = arr.shape[-1]
                return arr.reshape(voxels, channels)
            else:
                return arr.reshape(-1, 1)
        return arr.reshape(-1, 1)

    @staticmethod
    def _per_sample_zscore(M: np.ndarray) -> np.ndarray:
        mean = np.nanmean(M, axis=0, keepdims=True)
        std = np.nanstd(M, axis=0, keepdims=True)
        std = np.where(std == 0, 1.0, std)
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
                np.percentile(v, 1), np.percentile(v, 5),
                np.percentile(v, 95), np.percentile(v, 99),
                np.count_nonzero(v)
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

    # ------------------------
    # Fit / Transform
    # ------------------------
    def fit(self, X: pd.DataFrame, y=None):
        matrices = self._prepare_feature_matrix(X)
        
        # Replace NaN and inf with mean
        matrices = self._fill_invalid_with_row_mean(matrices)

        if self.scale_volume and self.strategy != "pca":
            self._scaler = StandardScaler()
            matrices = self._scaler.fit_transform(matrices)

        if self.strategy == "pca":
            if self.n_components is None:
                raise ValueError("n_components must be set for 'pca'.")
            self._scaler = StandardScaler()
            matrices_scaled = self._scaler.fit_transform(matrices)
            self._pca = PCA(n_components=self.n_components, random_state=self.random_state)
            # sanity check
            if not np.isfinite(matrices_scaled).all():
                raise ValueError("Non-finite values remain before PCA.")
            self._pca.fit(matrices_scaled)
            self.feature_names_ = [f"jmap_pca_{i:03d}" for i in range(self.n_components)]
        elif self.strategy == "flatten":
            self.feature_names_ = [f"jmap_flat_{i:07d}" for i in range(matrices.shape[1])]
        else:  # stats
            first_sample = self._to_2d(X.iloc[0][self.jmap_features[0]], self.keep_channel_axis)
            channels = first_sample.shape[1]
            stat_names = ("mean", "std", "min", "max", "median",
                          "p01", "p05", "p95", "p99", "nnz")
            names = []
            for jf in self.jmap_features:
                for ch in range(channels):
                    names += [f"{jf}_ch{ch}_{stat}" for stat in stat_names]
            self.feature_names_ = names

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        matrices = self._prepare_feature_matrix(X)
        
        # Replace NaN and inf with mean
        matrices = self._fill_invalid_with_row_mean(matrices)

        if self.scale_volume and self.strategy != "pca" and self._scaler is not None:
            matrices = self._scaler.transform(matrices)

        if self.strategy == "pca" and self._pca is not None:
            matrices_scaled = self._scaler.transform(matrices)
            matrices = self._pca.transform(matrices_scaled)
            


        return pd.DataFrame(
            matrices,
            index=X.index,
            columns=self.get_feature_names_out()
        )

    # ------------------------
    # Utility
    # ------------------------
    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_) if self.feature_names_ is not None else None

    def set_key_predictors(self, predictor_names: List[str]):
        """Store list of key predictor names (e.g., from downstream model coef_)."""
        self.key_predictors_ = predictor_names
        
    @staticmethod
    def _fill_invalid_with_row_mean(matrices: np.ndarray) -> np.ndarray:
        for i in range(matrices.shape[0]):
            row = matrices[i, :]
            mask = ~np.isfinite(row)
            if np.any(mask):
                valid_mean = np.nanmean(row)
                if np.isnan(valid_mean):
                    valid_mean = 0.0
                row[mask] = valid_mean
        return matrices

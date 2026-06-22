from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import numpy as np

class ScalePlusPCAPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=None, verbose=False):
        self.n_components = n_components
        self.verbose = verbose
        self._scaler = None
        self._pca = None
    # ------------------------
    # Fit / Transform
    # ------------------------
    def fit(self, X, y=None):
        if self.n_components is None:
            raise ValueError("n_components must be set for 'pca'.")
        self._scaler = StandardScaler()
        matrices_scaled = self._scaler.fit_transform(matrices)
        self._pca = PCA(n_components=self.n_components, random_state=self.random_state)
        # sanity check
        if not np.isfinite(matrices_scaled).all():
            raise ValueError("Non-finite values remain before PCA.")
        self._pca.fit(matrices_scaled)

    def fit_transform(self, X):
        matrices_scaled = self._scaler.transform(matrices)
        matrices = self._pca.transform(matrices_scaled)
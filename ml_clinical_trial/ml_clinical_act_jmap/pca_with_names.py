from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np

class PCAWithNames(BaseEstimator, TransformerMixin):
    """
    A thin wrapper around sklearn.decomposition.PCA that:
      1) Exposes a sklearn-compliant __init__ (no *args/**kwargs).
      2) Always returns a pandas DataFrame with columns f0..f{n_components_-1}.
      3) Provides get_feature_names_out() for downstream selectors.
    """
    def __init__(self,
                 n_components=None,
                 random_state=None):
        self.n_components = n_components
        self.random_state = random_state

        self.pca_ = None
        self.feature_names_out_ = None

    # ---- sklearn API ----
    def fit(self, X, y=None):
        self.pca_ = PCA(
            n_components=self.n_components,
            random_state=self.random_state,
        )
        self.pca_.fit(X, y)
        n_out = self.pca_.n_components_  # number of columns PCA will output
        self.feature_names_out_ = [f"f{i}" for i in range(n_out)]
        return self

    def transform(self, X):
        Z = self.pca_.transform(X)  # ndarray
        # preserve index if X is a DataFrame; else default RangeIndex
        idx = getattr(X, "index", None)
        return pd.DataFrame(Z, index=idx, columns=self.feature_names_out_)

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X, y).transform(X)

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_out_)

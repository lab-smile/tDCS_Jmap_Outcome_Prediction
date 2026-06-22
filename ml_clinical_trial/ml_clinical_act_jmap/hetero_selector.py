import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler

# Optional: scaler that preserves DataFrame structure
class DFStandardScaler(BaseEstimator, TransformerMixin):
    def __init__(self, with_mean=True, with_std=True):
        self.with_mean = with_mean
        self.with_std = with_std
        self._scaler = StandardScaler(with_mean=with_mean, with_std=with_std)

    def fit(self, X, y=None):
        Xv = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
        self._scaler.fit(Xv)
        self.columns_ = X.columns if isinstance(X, pd.DataFrame) else np.arange(Xv.shape[1])
        self.index_ = X.index if isinstance(X, pd.DataFrame) else None
        return self

    def transform(self, X):
        Xv = np.asarray(X) if not isinstance(X, pd.DataFrame) else X[self.input_columns_].values
        X_df = pd.DataFrame(Xv, index=getattr(X, "index", None), columns=self.input_columns_)
        return X_df.loc[:, self.selected_columns_]

# --- Stage 1: Welch t-test prefilter ---
class WelchTTestSelector(BaseEstimator, TransformerMixin):
    def __init__(self, p_thresh=1e-4, min_k_if_empty=2000, cap_after_t=15000):
        self.p_thresh = p_thresh
        self.min_k_if_empty = min_k_if_empty
        self.cap_after_t = cap_after_t

    def fit(self, X, y):
        from scipy.stats import ttest_ind
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(np.asarray(X), columns=[f"f{i}" for i in range(np.asarray(X).shape[1])])

        y = np.asarray(y).astype(int)
        if not set(np.unique(y)).issubset({0, 1}):
            raise ValueError("WelchTTestSelector expects binary labels {0,1}.")

        g0, g1 = (y == 0), (y == 1)
        if g0.sum() < 2 or g1.sum() < 2:
            raise ValueError("At least 2 samples per class required.")

        # Welch t-test, vectorized
        t_stat, pvals = ttest_ind(X.values[g0], X.values[g1], axis=0, equal_var=False, nan_policy='omit')
        pvals = np.where(np.isfinite(pvals), pvals, 1.0)

        sel = np.where(pvals < self.p_thresh)[0]
        if sel.size == 0:
            sel = np.argsort(pvals)[:min(self.min_k_if_empty, pvals.size)]
        if sel.size > self.cap_after_t:
            order = np.argsort(pvals[sel])
            sel = sel[order[:self.cap_after_t]]

        self.selected_mask_ = np.zeros(X.shape[1], dtype=bool)
        self.selected_mask_[sel] = True
        self.input_columns_ = X.columns.to_list()
        self.selected_columns_ = list(np.array(self.input_columns_)[self.selected_mask_])
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(np.asarray(X), columns=self.input_columns_)
        return X.loc[:, self.selected_columns_]

    def get_feature_names_out(self, input_features=None):
        return np.array(self.selected_columns_)

# --- Stage 2: mRMR on the prefiltered subset ---
class MRMRSelector(BaseEstimator, TransformerMixin):
    def __init__(self, frac_for_topk=0.01, min_topk=50, max_topk=1000, feature_prefix="jmap_zeyun"):
        self.frac_for_topk = frac_for_topk
        self.min_topk = min_topk
        self.max_topk = max_topk
        self.feature_prefix = feature_prefix

    def fit(self, X, y):
        try:
            from mrmr import mrmr_classif
        except Exception as e:
            raise ImportError("Install `mrmr-selection` (pip install mrmr-selection).") from e

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(np.asarray(X), columns=[f"f{i}" for i in range(np.asarray(X).shape[1])])

        # drop constant cols (helps small-n or degenerate cases)
        var = X.std(axis=0, ddof=0)
        Xnz = X.loc[:, var > 0]
        if Xnz.shape[1] == 0:
            raise ValueError("All features constant after prefilter; cannot run mRMR.")

        # K sizing
        rough = max(int(round(Xnz.shape[1] * self.frac_for_topk)), 1)
        K = min(max(rough, self.min_topk), self.max_topk, Xnz.shape[1])

        # run mRMR (MIQ by default)
        #chosen_cols = mrmr_classif(X=Xnz, y=pd.Series(np.asarray(y).astype(int)), K=K, show_progress=False)
        y_ser = pd.Series(np.asarray(y).ravel(), index=Xnz.index).astype(int)
        chosen_cols = mrmr_classif(X=Xnz, y=y_ser, K=K, show_progress=False)

        self.selected_columns_ = list(chosen_cols)
        # Compose nice names if you want a new prefix; otherwise keep original
        self.feature_names_out_ = [c if c in X.columns else f"{self.feature_prefix}_{c}" for c in self.selected_columns_]
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            # If upstream returned ndarray, we can't map by name — enforce DataFrame usage.
            raise TypeError("MRMRSelector expects a pandas DataFrame with column names from stage 1.")
        # Keep original column names (chosen_cols are from Xnz; they exist in X)
        return X.loc[:, self.selected_columns_]

    def get_feature_names_out(self, input_features=None):
        # return the *actual* column names that pass through
        return np.array(self.selected_columns_)

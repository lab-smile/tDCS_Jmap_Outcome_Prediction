# selectors.py
"""
Feature-selection stages:
  WelchTTestSelector  – fast Welch t-test pre-filter
  MRMRSelector        – mRMR on the pre-filtered subset
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1: Welch t-test pre-filter
# ─────────────────────────────────────────────────────────────────────────────

class WelchTTestSelector(BaseEstimator, TransformerMixin):
    """
    Retains only columns whose Welch t-test p-value < p_thresh
    (between class 0 and class 1).

    Parameters
    ----------
    p_thresh : float
        Significance threshold.
    min_k_if_empty : int
        Number of top features to fall back to when nothing passes p_thresh.
    cap_after_t : int
        Hard cap on the number of features forwarded to the next stage.
    """

    def __init__(
        self,
        p_thresh: float = 1e-4,
        min_k_if_empty: int = 2000,
        cap_after_t: int = 15000,
    ):
        self.p_thresh       = p_thresh
        self.min_k_if_empty = min_k_if_empty
        self.cap_after_t    = cap_after_t

    def fit(self, X, y):
        from scipy.stats import ttest_ind

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(
                np.asarray(X),
                columns=[f"f{i}" for i in range(np.asarray(X).shape[1])],
            )
        y = np.asarray(y).ravel()

        # ── For regression y is continuous; use median split ──────────────
        if not set(np.unique(y)).issubset({0, 1}):
            median = np.median(y)
            y_bin = (y > median).astype(int)
        else:
            y_bin = y.astype(int)

        g0, g1 = (y_bin == 0), (y_bin == 1)
        if g0.sum() < 2 or g1.sum() < 2:
            # Cannot run t-test – keep all
            sel = np.arange(X.shape[1])
        else:
            _, pvals = ttest_ind(
                X.values[g0], X.values[g1],
                axis=0, equal_var=False, nan_policy="omit",
            )
            pvals = np.where(np.isfinite(pvals), pvals, 1.0)
            sel   = np.where(pvals < self.p_thresh)[0]
            if sel.size == 0:
                sel = np.argsort(pvals)[: min(self.min_k_if_empty, pvals.size)]
            if sel.size > self.cap_after_t:
                order = np.argsort(pvals[sel])
                sel   = sel[order[: self.cap_after_t]]

        self.selected_mask_    = np.zeros(X.shape[1], dtype=bool)
        self.selected_mask_[sel] = True
        self.input_columns_      = X.columns.to_list()
        self.selected_columns_   = list(np.array(self.input_columns_)[self.selected_mask_])
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(np.asarray(X), columns=self.input_columns_)
        return X.loc[:, self.selected_columns_]

    def get_feature_names_out(self, input_features=None):
        return np.array(self.selected_columns_)


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2: mRMR selector
# ─────────────────────────────────────────────────────────────────────────────

class MRMRSelector(BaseEstimator, TransformerMixin):
    """
    Minimum Redundancy Maximum Relevance feature selection
    using the ``mrmr-selection`` package.

    Parameters
    ----------
    frac_for_topk : float
        Fraction of incoming features to request from mRMR.
    min_topk : int
        Lower bound on K.
    max_topk : int
        Upper bound on K (hyperparameter tuned by BO).
    feature_prefix : str
        Prefix used when renaming features (kept for compatibility).
    """

    def __init__(
        self,
        frac_for_topk: float = 0.01,
        min_topk: int = 50,
        max_topk: int = 1000,
        feature_prefix: str = "jmap_feat",
    ):
        self.frac_for_topk  = frac_for_topk
        self.min_topk       = min_topk
        self.max_topk       = max_topk
        self.feature_prefix = feature_prefix

    def fit(self, X, y):
        try:
            from mrmr import mrmr_classif, mrmr_regression
        except ImportError as e:
            raise ImportError(
                "Install mrmr-selection:  pip install mrmr-selection"
            ) from e

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(
                np.asarray(X),
                columns=[f"f{i}" for i in range(np.asarray(X).shape[1])],
            )
        y = np.asarray(y).ravel()

        # Drop constant columns
        var  = X.std(axis=0, ddof=0)
        Xnz  = X.loc[:, var > 0]
        if Xnz.shape[1] == 0:
            raise ValueError("All features constant after pre-filter.")

        rough = max(int(round(Xnz.shape[1] * self.frac_for_topk)), 1)
        K     = min(max(rough, self.min_topk), self.max_topk, Xnz.shape[1])

        y_ser = pd.Series(y, index=Xnz.index)

        # Use regression variant when y is continuous
        if set(np.unique(y)).issubset({0, 1}):
            chosen = mrmr_classif(
                X=Xnz, y=y_ser.astype(int), K=K, show_progress=False
            )
        else:
            chosen = mrmr_regression(
                X=Xnz, y=y_ser.astype(float), K=K, show_progress=False
            )

        self.selected_columns_  = list(chosen)
        self.feature_names_out_ = self.selected_columns_
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "MRMRSelector expects a pandas DataFrame with named columns."
            )
        return X.loc[:, self.selected_columns_]

    def get_feature_names_out(self, input_features=None):
        return np.array(self.selected_columns_)
# pipeline_factory.py
"""
Builds imblearn Pipelines for classification and regression.

Classification estimator : SGDClassifier  (log_loss)
Regression estimator     : SGDRegressor

Both are preceded by:
  JmapACTPreprocessor  ->  SafeSMOTE (clf only)  ->  PCAWithNames
  ->  WelchTTestSelector  ->  MRMRSelector
  ->  RBFSampler  ->  SGD{Classifier|Regressor}
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from imblearn.pipeline import Pipeline
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDClassifier, SGDRegressor
from sklearn.svm import SVR 

from constants import ATLAS_PATH, ATLAS_LABELS_PATH
from preprocessors import (
    JmapACTPreprocessor,
    PCAWithNames,
    SafeSMOTE,
)
from feature_selectors import MRMRSelector, WelchTTestSelector


def _get_jmap_cols(input_config: str) -> list[str]:
    """Return the jmap column names expected in the DataFrame."""
    mapping = {
        "magnitude":           ["jmap_mag"],
        "theta_phi":           ["jmap_theta", "jmap_phi"],
        "magnitude_theta_phi": ["jmap_mag", "jmap_theta", "jmap_phi"],
        "jxyz":                ["jmap_jx", "jmap_jy", "jmap_jz"],
        "magnitude_demo":      ["jmap_mag"],
    }
    return mapping[input_config]


def build_clf_pipeline(
    input_config: str,
    random_state: int = 42,
) -> Pipeline:
    """Return a fresh classification pipeline for the given input config."""
    jmap_cols = _get_jmap_cols(input_config)

    return Pipeline([
        ("prep", JmapACTPreprocessor(
            jmap_features=jmap_cols,
            strategy="flatten",
            keep_channel_axis=True,
            random_state=random_state,
            atlas_path=ATLAS_PATH,
            atlas_labels_path=ATLAS_LABELS_PATH,
            scale_volume=True,
        )),
        # ("smote", SafeSMOTE(
        #     sampling_strategy=1.0,
        #     k_neighbors=2,
        #     random_state=random_state,
        # )),
        ("pca", PCAWithNames(n_components=0.25, random_state=random_state)),
        ("ttest", WelchTTestSelector(
            p_thresh=1e-4,
            min_k_if_empty=2000,
            cap_after_t=15000,
        )),
        ("mrmr", MRMRSelector(
            frac_for_topk=0.01,
            min_topk=10,
            max_topk=20,
        )),
        ("rbf", RBFSampler(
            gamma=1.0,
            n_components=10,
            random_state=random_state,
        )),
        ("sgd", SGDClassifier(
            loss="log_loss",
            max_iter=1000,
            random_state=random_state,
        )),
    ])


def build_reg_pipeline(
    input_config: str,
    random_state: int = 42,
) -> Pipeline:
    """Return a fresh regression pipeline for the given input config."""
    jmap_cols = _get_jmap_cols(input_config)

    return Pipeline([
        ("prep", JmapACTPreprocessor(
            jmap_features=jmap_cols,
            strategy="flatten",
            keep_channel_axis=True,
            random_state=random_state,
            atlas_path=ATLAS_PATH,
            atlas_labels_path=ATLAS_LABELS_PATH,
            scale_volume=True,
        )),
        # No SMOTE for regression
        ("pca", PCAWithNames(n_components=0.95, random_state=random_state)),
        ("ttest", WelchTTestSelector(
            p_thresh=1e-4,
            min_k_if_empty=2000,
            cap_after_t=15000,
        )),
        ("mrmr", MRMRSelector(
            frac_for_topk=0.01,
            min_topk=10,
            max_topk=20,
        )),
        ("rbf", RBFSampler(
            gamma=1.0,
            n_components=10,
            random_state=random_state,
        )),
        ("svr", SVR(
            kernel="rbf",
            C=1.0,
            epsilon=0.1,
        )),
    ])


def attach_demo_features(
    X_jmap: pd.DataFrame,
    demo_df: pd.DataFrame,
    input_config: str,
) -> pd.DataFrame:
    """
    For input_config == 'magnitude_demo', append numeric demographic
    columns to the jmap DataFrame (aligned by subject_id index).
    For all other configs, return X_jmap unchanged.
    """
    if input_config != "magnitude_demo":
        return X_jmap

    numeric_demo = demo_df.select_dtypes(include=[np.number]).drop(
        columns=["responder", "stai_state_tp1", "stai_state_tp2",
                 "stai_state_decrease"],
        errors="ignore",
    )
    # align on index
    shared = X_jmap.index.intersection(numeric_demo.index)
    demo_aligned = numeric_demo.loc[shared].add_prefix("demo_")
    return pd.concat([X_jmap.loc[shared], demo_aligned], axis=1)
"""
Metrics and uncertainty estimation (bootstrap).

We compute:
- AUROC
- AUPRC (Average Precision)
- Balanced Accuracy
- Weighted F1
- PPV (Precision)
- TPR (Recall)
- MCC

For mean ± SD and 95% CI, we bootstrap over the *test subjects*.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, Callable, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    matthews_corrcoef,
)

METRIC_ORDER = [
    "AUPRC",
    "AUROC",
    "Balanced Accuracy",
    "Weighted F1",
    "PPV",
    "TPR",
    "MCC",
]

def summarize_repetition_metrics(rep_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric in [
        "AUROC",
        "AUPRC",
        "BalancedAccuracy",
        "WeightedF1",
        "MCC",
    ]:
        rows.append({
            "Metric": metric,
            "Mean": rep_df[metric].mean(),
            "Std":  rep_df[metric].std(ddof=1),
        })
    return pd.DataFrame(rows)



def _safe_auc(metric_fn: Callable, y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    AUC metrics require both classes in y_true. Return NaN if not computable.
    """
    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        return np.nan
    return float(metric_fn(y_true, y_score))


def compute_point_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    out = {}
    out["AUPRC"] = _safe_auc(average_precision_score, y_true, y_prob)
    out["AUROC"] = _safe_auc(roc_auc_score, y_true, y_prob)
    out["Balanced Accuracy"] = float(balanced_accuracy_score(y_true, y_pred))
    out["Weighted F1"] = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    out["PPV"] = float(precision_score(y_true, y_pred, zero_division=0))
    out["TPR"] = float(recall_score(y_true, y_pred, zero_division=0))
    out["MCC"] = float(matthews_corrcoef(y_true, y_pred)) if len(np.unique(y_true)) > 1 else np.nan
    return out


@dataclass
class BootstrapSummary:
    mean: float
    sd: float
    ci_low: float
    ci_high: float


def bootstrap_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_boot: int = 2000,
    seed: int = 42,
    threshold: float = 0.5,
) -> Dict[str, BootstrapSummary]:
    """
    Bootstrap over subjects (rows) to estimate distribution of each metric.

    For AUROC/AUPRC, resamples that end up single-class are skipped (NaN) and ignored.
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    n = len(y_true)
    if n != len(y_prob):
        raise ValueError("y_true and y_prob must have same length")

    samples: Dict[str, List[float]] = {k: [] for k in METRIC_ORDER}

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)  # resample with replacement
        m = compute_point_metrics(y_true[idx], y_prob[idx], threshold=threshold)
        for k in METRIC_ORDER:
            v = m[k]
            if np.isnan(v):
                continue
            samples[k].append(v)

    out: Dict[str, BootstrapSummary] = {}
    for k in METRIC_ORDER:
        arr = np.asarray(samples[k], dtype=float)
        if arr.size == 0:
            out[k] = BootstrapSummary(mean=np.nan, sd=np.nan, ci_low=np.nan, ci_high=np.nan)
            continue
        mean = float(np.mean(arr))
        sd = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
        ci_low, ci_high = np.percentile(arr, [2.5, 97.5]).astype(float)
        out[k] = BootstrapSummary(mean=mean, sd=sd, ci_low=float(ci_low), ci_high=float(ci_high))
    return out

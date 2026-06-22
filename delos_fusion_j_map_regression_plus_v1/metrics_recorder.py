# metrics_recorder.py
"""
Helpers to compute and accumulate per-repetition metrics.

Classification
--------------
  AUROC, AUPRC, Balanced Accuracy, Weighted F1, MCC

Regression
----------
  MAE, RMSE, R²
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)


# ─────────────────────────────────────────────────────────────────────────────
# Classification
# ─────────────────────────────────────────────────────────────────────────────

def compute_clf_metrics(y_true, y_pred, y_prob) -> Dict[str, float]:
    """
    Parameters
    ----------
    y_true : array of {0,1}
    y_pred : array of {0,1}   (hard labels)
    y_prob : array of float   (probability for class 1)
    """
    try:
        auroc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auroc = np.nan

    try:
        auprc = average_precision_score(y_true, y_prob)
    except ValueError:
        auprc = np.nan

    return {
        "AUROC":             auroc,
        "AUPRC":             auprc,
        "Balanced Accuracy": balanced_accuracy_score(y_true, y_pred),
        "Weighted F1":       f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "MCC":               matthews_corrcoef(y_true, y_pred),
    }


def summarise_clf_metrics(records: List[Dict]) -> pd.DataFrame:
    """
    Given a list of per-repetition metric dicts, return a one-row DataFrame
    with 'metric  mean ± SD' formatted strings.
    """
    df   = pd.DataFrame(records)
    rows = []
    for col in ["AUROC", "AUPRC", "Balanced Accuracy", "Weighted F1", "MCC"]:
        vals = df[col].dropna()
        rows.append({
            "Metric": col,
            "Mean":   round(vals.mean(), 4),
            "SD":     round(vals.std(ddof=1), 4),
            "Mean ± SD": f"{vals.mean():.4f} ± {vals.std(ddof=1):.4f}",
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Regression
# ─────────────────────────────────────────────────────────────────────────────

def compute_reg_metrics(y_true, y_pred) -> Dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    return {
        "MAE":  mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mse),
        "R2":   r2_score(y_true, y_pred),
    }


def summarise_reg_metrics(records: List[Dict]) -> pd.DataFrame:
    df   = pd.DataFrame(records)
    rows = []
    for col, label in [("MAE", "MAE"), ("RMSE", "RMSE"), ("R2", "R²")]:
        vals = df[col].dropna()
        rows.append({
            "Metric":    label,
            "Mean":      round(vals.mean(), 4),
            "SD":        round(vals.std(ddof=1), 4),
            "Mean ± SD": f"{vals.mean():.4f} ± {vals.std(ddof=1):.4f}",
        })
    return pd.DataFrame(rows)
"""
Plot ROC and PR curves with bootstrap confidence bands (test set).

We compute curves for:
- the point estimate from all test subjects
- bootstrap resamples to derive a 95% CI band for TPR at fixed FPR (ROC) or Precision at fixed Recall (PR)

Outputs are saved as PNG, SVG, and PDF.
"""

from __future__ import annotations
from typing import Tuple, Optional, Dict

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, auc

def _bootstrap_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    kind: str,
    n_boot: int = 2000,
    seed: int = 42,
    grid_size: int = 200,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
        x_grid, y_low, y_high (95% band)
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    n = len(y_true)

    if kind == "roc":
        x_grid = np.linspace(0, 1, grid_size)
        ys = []
        for _ in range(n_boot):
            idx = rng.integers(0, n, size=n)
            if len(np.unique(y_true[idx])) < 2:
                continue
            fpr, tpr, _ = roc_curve(y_true[idx], y_prob[idx])
            # interpolate tpr over fixed fpr grid
            tpr_i = np.interp(x_grid, fpr, tpr, left=0, right=1)
            ys.append(tpr_i)
    elif kind == "pr":
        x_grid = np.linspace(0, 1, grid_size)  # recall grid
        ys = []
        for _ in range(n_boot):
            idx = rng.integers(0, n, size=n)
            if len(np.unique(y_true[idx])) < 2:
                continue
            prec, rec, _ = precision_recall_curve(y_true[idx], y_prob[idx])
            # precision_recall_curve returns recall sorted ascending, but precision length matches.
            # interpolate precision over recall grid (note: rec is increasing)
            # ensure monotonic bounds for interpolation
            order = np.argsort(rec)
            rec_s = rec[order]
            prec_s = prec[order]
            # handle duplicate recall values
            rec_u, idx_u = np.unique(rec_s, return_index=True)
            prec_u = prec_s[idx_u]
            prec_i = np.interp(x_grid, rec_u, prec_u, left=prec_u[0], right=prec_u[-1])
            ys.append(prec_i)
    else:
        raise ValueError("kind must be 'roc' or 'pr'")

    if len(ys) == 0:
        return x_grid, np.full_like(x_grid, np.nan), np.full_like(x_grid, np.nan)

    ys = np.asarray(ys)
    y_low = np.percentile(ys, 2.5, axis=0)
    y_high = np.percentile(ys, 97.5, axis=0)
    return x_grid, y_low, y_high


def plot_roc(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    out_png: str,
    out_svg: str,
    out_pdf: str,
    auc_value: Optional[float] = None,
    n_boot: int = 2000,
    seed: int = 42,
):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    if len(np.unique(y_true)) < 2:
        raise ValueError("ROC requires both classes in y_true")

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    if auc_value is None:
        auc_value = auc(fpr, tpr)

    xg, lo, hi = _bootstrap_curves(y_true, y_prob, kind="roc", n_boot=n_boot, seed=seed)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.plot([0,1], [0,1], linestyle="--", linewidth=1.5, label="Chance")
    ax.plot(fpr, tpr, linewidth=2.5, label=f"Mean ROC (AUC = {auc_value:.2f})")
    ax.fill_between(xg, lo, hi, alpha=0.2, label="95% CI")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right", frameon=True)
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_svg)
    fig.savefig(out_pdf)
    plt.close(fig)


def plot_pr(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    out_png: str,
    out_svg: str,
    out_pdf: str,
    ap_value: Optional[float] = None,
    n_boot: int = 2000,
    seed: int = 42,
):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    if len(np.unique(y_true)) < 2:
        raise ValueError("PR requires both classes in y_true")

    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    if ap_value is None:
        # Average precision (sklearn) is a summary; AUC(rec,prec) differs slightly.
        from sklearn.metrics import average_precision_score
        ap_value = float(average_precision_score(y_true, y_prob))

    base_rate = float(np.mean(y_true))  # chance line in PR
    xg, lo, hi = _bootstrap_curves(y_true, y_prob, kind="pr", n_boot=n_boot, seed=seed)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.hlines(base_rate, 0, 1, linestyles="--", linewidth=1.5, label="Chance")
    ax.plot(rec, prec, linewidth=2.5, label=f"Mean PR (AP = {ap_value:.2f})")
    ax.fill_between(xg, lo, hi, alpha=0.2, label="95% CI")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(loc="lower left", frameon=True)
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_svg)
    fig.savefig(out_pdf)
    plt.close(fig)

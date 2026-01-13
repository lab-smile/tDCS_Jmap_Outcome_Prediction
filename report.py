"""
Report generation:
- metrics CSV
- summary PDF (IDs, labels, predicted probabilities, performance table, ROC+PR figures)

This module expects that ROC/PR figures are already generated as images.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from .metrics import METRIC_ORDER, BootstrapSummary


def metrics_to_dataframe(bs: Dict[str, BootstrapSummary]) -> pd.DataFrame:
    rows = []
    for k in METRIC_ORDER:
        s = bs[k]
        rows.append({
            "Metric": k,
            "Mean": s.mean,
            "SD": s.sd,
            "95% CI Lower": s.ci_low,
            "95% CI Upper": s.ci_high,
            "Mean ± SD": f"{s.mean:.2f} ± {s.sd:.2f}" if np.isfinite(s.mean) else "NA",
            "95% CI (Lower – Upper)": f"{s.ci_low:.2f} – {s.ci_high:.2f}" if np.isfinite(s.ci_low) else "NA",
        })
    df = pd.DataFrame(rows)
    df = df[["Metric", "Mean ± SD", "95% CI (Lower – Upper)", "Mean", "SD", "95% CI Lower", "95% CI Upper"]]
    return df


def save_metrics_csv(df: pd.DataFrame, out_csv: str):
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)


def _subject_table(ids: List[str], y_true: List[int], y_prob: Optional[np.ndarray] = None) -> pd.DataFrame:
    d = {"subject_id": ids, "y_true": list(map(int, y_true))}
    if y_prob is not None:
        d["y_prob"] = list(map(float, y_prob))
        d["y_pred"] = list(map(int, (np.asarray(y_prob) >= 0.5).astype(int)))
    return pd.DataFrame(d)


def make_summary_pdf(
    out_pdf: str,
    train_ids: List[str],
    train_y: List[int],
    test_ids: List[str],
    test_y: List[int],
    test_prob: np.ndarray,
    metrics_df: pd.DataFrame,
    roc_img_path: str,
    pr_img_path: str,
    title: str = "Direction-ML: Test-set Performance Summary",
    notes: Optional[List[str]] = None,
):
    """
    Creates a single PDF with:
    - title
    - train/test subject tables
    - performance table
    - ROC and PR plots (as images embedded)
    """
    Path(out_pdf).parent.mkdir(parents=True, exist_ok=True)

    train_tbl = _subject_table(train_ids, train_y)
    test_tbl  = _subject_table(test_ids, test_y, y_prob=test_prob)

    with PdfPages(out_pdf) as pdf:
        # Page 1: Title + notes
        fig = plt.figure(figsize=(8.5, 11))
        fig.text(0.08, 0.93, title, fontsize=16, weight="bold")
        fig.text(0.08, 0.90, "Train/Test splits, labels, and test-set metrics with 95% CI.", fontsize=10)

        y = 0.86
        if notes:
            fig.text(0.08, y, "Notes:", fontsize=11, weight="bold")
            y -= 0.02
            for n in notes[:10]:
                fig.text(0.10, y, f"• {n}", fontsize=9)
                y -= 0.018

        fig.text(0.08, 0.60, "Performance (Test Set)", fontsize=12, weight="bold")
        # render metrics table
        ax = fig.add_axes([0.08, 0.40, 0.84, 0.18])
        ax.axis("off")
        base_cols = ["Metric", "Mean ± SD", "95% CI (Lower – Upper)"]
        if "Level" not in metrics_df.columns:
            table_df = metrics_df[base_cols]
            tbl = ax.table(
                cellText=table_df.values,
                colLabels=table_df.columns,
                loc="center",
                cellLoc="left",
                colLoc="left",
            )
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(9)
            tbl.scale(1, 1.3)
        else:
            for level_name in metrics_df["Level"].unique():
                sub = metrics_df[metrics_df["Level"] == level_name][base_cols]
                table_df = metrics_df[base_cols]
                tbl = ax.table(
                    cellText=table_df.values,
                    colLabels=table_df.columns,
                    loc="center",
                    cellLoc="left",
                    colLoc="left",
                )
                tbl.auto_set_font_size(False)
                tbl.set_fontsize(9)
                tbl.scale(1, 1.3)

        fig.text(0.08, 0.34, "Train Set Subjects", fontsize=12, weight="bold")
        ax2 = fig.add_axes([0.08, 0.20, 0.84, 0.12])
        ax2.axis("off")
        tdf = train_tbl
        tdf2 = tdf.copy()
        tdf2["site"] = ["UF" if sid.startswith(("1","2")) else "UA" for sid in tdf2["subject_id"]]
        tdf2 = tdf2[["subject_id","site","y_true"]]
        tbl2 = ax2.table(
            cellText=tdf2.values,
            colLabels=tdf2.columns,
            loc="center",
            cellLoc="left",
            colLoc="left",
        )
        tbl2.auto_set_font_size(False)
        tbl2.set_fontsize(8)
        tbl2.scale(1, 1.2)

        fig.text(0.08, 0.16, "Test Set Subjects (with predictions)", fontsize=12, weight="bold")
        ax3 = fig.add_axes([0.08, 0.03, 0.84, 0.12])
        ax3.axis("off")
        sdf = test_tbl.copy()
        sdf["site"] = ["UF" if sid.startswith(("1","2")) else "UA" for sid in sdf["subject_id"]]
        sdf = sdf[["subject_id","site","y_true","y_prob","y_pred"]]
        # format probabilities to 3 decimals for readability
        cell = sdf.copy()
        cell["y_prob"] = cell["y_prob"].map(lambda x: f"{x:.3f}")
        tbl3 = ax3.table(
            cellText=cell.values,
            colLabels=cell.columns,
            loc="center",
            cellLoc="left",
            colLoc="left",
        )
        tbl3.auto_set_font_size(False)
        tbl3.set_fontsize(8)
        tbl3.scale(1, 1.2)

        pdf.savefig(fig)
        plt.close(fig)

        # Page 2: Curves
        fig = plt.figure(figsize=(8.5, 11))
        fig.text(0.08, 0.95, "Test-set ROC and PR Curves", fontsize=14, weight="bold")

        # embed images
        roc_img = plt.imread(roc_img_path)
        pr_img  = plt.imread(pr_img_path)

        ax1 = fig.add_axes([0.10, 0.55, 0.80, 0.35])
        ax1.imshow(roc_img)
        ax1.axis("off")
        ax2 = fig.add_axes([0.10, 0.12, 0.80, 0.35])
        ax2.imshow(pr_img)
        ax2.axis("off")

        pdf.savefig(fig)
        plt.close(fig)

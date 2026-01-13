"""
Main runner.

Example:
    python -m direction_ml.run_experiment

What it does:
1) Loads MNI-warped theta/phi volumes for train + test IDs
2) Wraps volumes into the input format expected by your JmapACTPreprocessor
3) Tunes hyperparameters using repeated 3-fold CV within training set
4) Fits best model and evaluates on test set
5) Saves:
   - ROC curve: PNG/SVG/PDF
   - PR curve:  PNG/SVG/PDF
   - metrics table: CSV
   - summary report: PDF
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Any
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from .config import (
    OLD_WORKING_DIR, NEW_WORKING_DIR,
    TRAIN_IDS_RESPONDER, TRAIN_IDS_NONRESP,
    TEST_IDS_RESPONDER, TEST_IDS_NONRESP,
    RANDOM_STATE, HYPEROPT_STRATEGY, N_SPLITS, N_REPEATS,
    BOOTSTRAP_N, BOOTSTRAP_SEED,
    OUTDIR_NAME,
    ROC_PNG, ROC_SVG, ROC_PDF,
    PR_PNG, PR_SVG, PR_PDF,
    METRICS_CSV, SUMMARY_PDF,
    SubjectSplit,
    FEATURE_MODE,
    SPLIT_MODE, CROSS_SITE_TEST_SITE, OUTER_N_SPLITS, REPORT_EVERY_SEC,
    ALL_IDS_RESPONDER, ALL_IDS_NONRESP,
)
from .data_io import load_dataset
from .train_eval import nested_outer_cv_fit_and_predict
from .metrics import bootstrap_metrics, summarize_repetition_metrics
from .plots import plot_roc, plot_pr
from .report import metrics_to_dataframe, save_metrics_csv, make_summary_pdf
from .features import as_jmap_dataframe
from .train_eval import repeated_nested_cv_oof, add_ci_from_repetitions

def infer_site(subject_id: str) -> str:
    # consistent with your notes/report logic
    return "UF" if str(subject_id).startswith(("1", "2")) else "UA"

def build_subject_split():
    if SPLIT_MODE == "manual":
        return SubjectSplit.from_lists(
            train_resp=TRAIN_IDS_RESPONDER,
            train_non=TRAIN_IDS_NONRESP,
            test_resp=TEST_IDS_RESPONDER,
            test_non=TEST_IDS_NONRESP,
        )

    # pooled
    all_ids = list(ALL_IDS_RESPONDER) + list(ALL_IDS_NONRESP)
    all_y   = [1]*len(ALL_IDS_RESPONDER) + [0]*len(ALL_IDS_NONRESP)

    if SPLIT_MODE == "mixed_site":
        # No fixed test set in the new mixed-site design
        return SubjectSplit(train_ids=all_ids, train_y=all_y, test_ids=[], test_y=[])

    if SPLIT_MODE == "cross_site":
        te_ids, te_y, tr_ids, tr_y = [], [], [], []
        for sid, yy in zip(all_ids, all_y):
            if infer_site(sid) == CROSS_SITE_TEST_SITE:
                te_ids.append(sid); te_y.append(int(yy))
            else:
                tr_ids.append(sid); tr_y.append(int(yy))
        return SubjectSplit(train_ids=tr_ids, train_y=tr_y, test_ids=te_ids, test_y=te_y)

    raise ValueError(f"Unknown SPLIT_MODE={SPLIT_MODE}")

def summarize_with_ci(df: pd.DataFrame, metric_cols, level: str, alpha: float = 0.05) -> pd.DataFrame:
    """
    Summarize metrics in df as mean±SD and percentile CI.

    CI is computed as empirical quantiles over the rows of df:
      [alpha/2, 1-alpha/2] -> default 95% CI.

    Returns columns required by report.py:
      Metric, Mean ± SD, 95% CI (Lower – Upper), Mean, SD, 95% CI Lower, 95% CI Upper
    Plus: Level (so we can show both summaries in the PDF).
    """
    rows = []
    for m in metric_cols:
        x = pd.to_numeric(df[m], errors="coerce").dropna().to_numpy()
        if x.size == 0:
            continue

        mean = float(np.mean(x))
        sd   = float(np.std(x, ddof=1)) if x.size > 1 else 0.0
        lo   = float(np.quantile(x, alpha / 2))
        hi   = float(np.quantile(x, 1 - alpha / 2))

        rows.append({
            "Level": level,
            "Metric": m,
            "Mean ± SD": f"{mean:.3f} ± {sd:.3f}",
            "95% CI (Lower – Upper)": f"{lo:.3f} – {hi:.3f}",
            "Mean": mean,
            "SD": sd,
            "95% CI Lower": lo,
            "95% CI Upper": hi,
        })

    return pd.DataFrame(rows)

def main():
    """
    Entry point for experiments.

    Outputs (same as earlier version):
      - summary_report.pdf
      - test_metrics.csv
      - test_roc.(png/svg/pdf)
      - test_pr.(png/svg/pdf)

    SPLIT_MODE behavior:
      - "manual" / "cross_site": fixed held-out test set (original behavior)
      - "mixed_site": BOOTSTRAP_N repetitions of OUTER_N_SPLITS-fold nested stratified CV on pooled UF+UA
                     (no fixed held-out test). We generate "test_*" outputs using mean OOF probs.
                     Also writes: repetition_metrics.csv (per-repetition metrics).
    """
    from pathlib import Path
    from datetime import datetime
    import numpy as np
    import pandas as pd

    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    outdir_name = OUTDIR_NAME + timestamp
    outdir = Path(NEW_WORKING_DIR) / outdir_name
    outdir.mkdir(parents=True, exist_ok=True)

    # snapshot config for reproducibility (best-effort)
    try:
        import inspect
        import direction_ml_mixed.config as _cfg_mod  # adjust if your package name differs
        (outdir / "config.py").write_text(inspect.getsource(_cfg_mod), encoding="utf-8")
    except Exception:
        pass

    split = build_subject_split()

    # ---------------------------
    # MIXED-SITE (repeated nested CV; no fixed test set)
    # ---------------------------
    if SPLIT_MODE == "mixed_site":
        # Load pooled dataset (all subjects participate in CV)
        ds = load_dataset(split.train_ids, split.train_y, old_working_dir=OLD_WORKING_DIR)
        X_df = as_jmap_dataframe(ds.X, ds.ids, mode=FEATURE_MODE)
        y = np.asarray(ds.y, dtype=int)

        # Run BOOTSTRAP_N repetitions of OUTER_N_SPLITS-fold nested stratified CV
        # This function should:
        #   - return per-rep metric dicts including:
        #       AUROC, AUPRC, BalancedAccuracy, WeightedF1, MCC (+rep)
        #   - return oof_prob_mean: mean OOF probability per subject across repetitions
        #   - return diagnostics dict
        rep_metrics, fold_metrics, oof_prob_mean, diagnostics = repeated_nested_cv_oof(
            X=X_df,
            y=y,
            strategy=HYPEROPT_STRATEGY,
            random_state=RANDOM_STATE,
            outer_n_splits=OUTER_N_SPLITS,
            inner_n_splits=N_SPLITS,
            inner_n_repeats=N_REPEATS,
            n_reps=BOOTSTRAP_N,
            report_every_sec=REPORT_EVERY_SEC,
        )

        rep_df  = pd.DataFrame(rep_metrics)
        fold_df = pd.DataFrame(fold_metrics)
        
        rep_df.to_csv(outdir / "repetition_metrics.csv", index=False)
        fold_df.to_csv(outdir / "outer_fold_metrics.csv", index=False)

        metric_cols = ["AUROC", "AUPRC", "BalancedAccuracy", "WeightedF1", "MCC"]
    
        # Rep-level: one metric value per repetition (OOF pooled)
        summary_rep = summarize_with_ci(
            rep_df,
            metric_cols=metric_cols,
            level="Repetition (OOF pooled)",
            alpha=0.05,
        )

        # Fold-level: one metric value per outer fold test, across all reps × folds
        summary_fold = summarize_with_ci(
            fold_df,
            metric_cols=metric_cols,
            level="Outer-fold test (all reps × folds)",
            alpha=0.05,
        )
        summary_fold["Level"] = "Outer-fold test (all reps × folds)"

        summary_all = pd.concat([summary_rep, summary_fold], ignore_index=True)
        # Save as the canonical CSV used by report.py
        summary_all.to_csv(outdir / METRICS_CSV, index=False)

        # Curves (use mean OOF prob per subject)
        auc_mean = float(rep_df["AUROC"].mean()) if "AUROC" in rep_df.columns else None
        ap_mean = float(rep_df["AUPRC"].mean()) if "AUPRC" in rep_df.columns else None

        plot_roc(
            y_true=y,
            y_prob=oof_prob_mean,
            out_png=str(outdir / ROC_PNG),
            out_svg=str(outdir / ROC_SVG),
            out_pdf=str(outdir / ROC_PDF),
            auc_value=auc_mean if (auc_mean is not None and np.isfinite(auc_mean)) else None,
            n_boot=BOOTSTRAP_N,  # used only for CI band in plotting (if implemented that way)
            seed=BOOTSTRAP_SEED,
        )
        plot_pr(
            y_true=y,
            y_prob=oof_prob_mean,
            out_png=str(outdir / PR_PNG),
            out_svg=str(outdir / PR_SVG),
            out_pdf=str(outdir / PR_PDF),
            ap_value=ap_mean if (ap_mean is not None and np.isfinite(ap_mean)) else None,
            n_boot=BOOTSTRAP_N,
            seed=BOOTSTRAP_SEED,
        )

        notes = [
            f"Old workspace (inputs): {OLD_WORKING_DIR}",
            f"New workspace (outputs): {NEW_WORKING_DIR}",
            f"Hyperparameter strategy: {HYPEROPT_STRATEGY}",
            f"Mixed-site design: {BOOTSTRAP_N} repetitions of {OUTER_N_SPLITS}-fold nested stratified CV (pooled UF+UA)",
            f"Inner CV: RepeatedStratifiedKFold (n_splits={N_SPLITS}, n_repeats={N_REPEATS})",
            "Site mapping: IDs starting with 1 or 2 = UF; IDs starting with 3 = UA",
            "Curves use mean out-of-fold (OOF) probability per subject across repetitions.",
        ]

        # Summary PDF:
        # - We keep the same output filename (SUMMARY_PDF)
        # - We pass 'test_prob' as the mean OOF probability vector (interpretation differs from held-out test)
        # - metrics_df now includes BOTH repetition-level and outer-fold-level summaries
        make_summary_pdf(
            out_pdf=str(outdir / SUMMARY_PDF),
            train_ids=list(ds.ids),
            train_y=list(map(int, y)),
            test_ids=list(ds.ids),
            test_y=list(map(int, y)),
            test_prob=oof_prob_mean,
            metrics_df=summary_all,   # <-- contains Mean±SD and CI for BOTH levels
            roc_img_path=str(outdir / ROC_PNG),
            pr_img_path=str(outdir / PR_PNG),
            notes=notes + [
                "NOTE: In mixed_site, the 'test' table contains out-of-fold (OOF) predictions, not a held-out test set.",
                "Metrics are reported at two levels: (1) repetition-level OOF pooled and (2) outer-fold test across all reps × folds.",
                "95% CIs are percentile intervals computed over repetitions or over all outer-fold evaluations, respectively.",
            ],
        )


        print("\n=== DONE (mixed_site repeated nested CV) ===")
        print(f"Outputs written to: {outdir}")
        print(f"Summary: {outdir / SUMMARY_PDF}")
        return

    # ---------------------------
    # MANUAL / CROSS-SITE (fixed held-out test set; original behavior)
    # ---------------------------
    train_ds = load_dataset(split.train_ids, split.train_y, old_working_dir=OLD_WORKING_DIR)
    test_ds = load_dataset(split.test_ids, split.test_y, old_working_dir=OLD_WORKING_DIR)

    X_train_df = as_jmap_dataframe(train_ds.X, train_ds.ids, mode=FEATURE_MODE)
    X_test_df = as_jmap_dataframe(test_ds.X, test_ds.ids, mode=FEATURE_MODE)

    y_train = np.asarray(train_ds.y, dtype=int)
    y_test = np.asarray(test_ds.y, dtype=int)

    train_res, y_prob_test, diagnostics = nested_outer_cv_fit_and_predict(
        X_train=X_train_df,
        y_train=y_train,
        X_test=X_test_df,
        strategy=HYPEROPT_STRATEGY,
        random_state=RANDOM_STATE,
        outer_n_splits=OUTER_N_SPLITS,
        inner_n_splits=N_SPLITS,
        inner_n_repeats=N_REPEATS,
        report_every_sec=REPORT_EVERY_SEC,
    )

    # Bootstrap metrics on held-out test set (original behavior)
    bs = bootstrap_metrics(
        y_true=y_test,
        y_prob=y_prob_test,
        n_boot=BOOTSTRAP_N,
        seed=BOOTSTRAP_SEED,
        threshold=0.5,
    )
    mdf = metrics_to_dataframe(bs)
    save_metrics_csv(mdf, str(outdir / METRICS_CSV))

    # Curves (test set)
    auc_mean = bs["AUROC"].mean
    ap_mean = bs["AUPRC"].mean

    plot_roc(
        y_true=y_test,
        y_prob=y_prob_test,
        out_png=str(outdir / ROC_PNG),
        out_svg=str(outdir / ROC_SVG),
        out_pdf=str(outdir / ROC_PDF),
        auc_value=auc_mean if np.isfinite(auc_mean) else None,
        n_boot=BOOTSTRAP_N,
        seed=BOOTSTRAP_SEED,
    )
    plot_pr(
        y_true=y_test,
        y_prob=y_prob_test,
        out_png=str(outdir / PR_PNG),
        out_svg=str(outdir / PR_SVG),
        out_pdf=str(outdir / PR_PDF),
        ap_value=ap_mean if np.isfinite(ap_mean) else None,
        n_boot=BOOTSTRAP_N,
        seed=BOOTSTRAP_SEED,
    )

    notes = [
        f"Old workspace (inputs): {OLD_WORKING_DIR}",
        f"New workspace (outputs): {NEW_WORKING_DIR}",
        f"Hyperparameter strategy: {HYPEROPT_STRATEGY}",
        f"CV: RepeatedStratifiedKFold (n_splits={N_SPLITS}, n_repeats={N_REPEATS})",
        f"Best CV ROC-AUC: {train_res.cv_best_score:.3f}",
        f"Best params: {train_res.best_params}",
        f"Outer CV (train split) AUC: {diagnostics['outer_auc_mean']:.3f} ± {diagnostics['outer_auc_std']:.3f}",
        "Site mapping: IDs starting with 1 or 2 = UF; IDs starting with 3 = UA",
        "Theta/Phi are treated as channels (C=2) and flattened by the preprocessor (unless you change strategy).",
    ]

    make_summary_pdf(
        out_pdf=str(outdir / SUMMARY_PDF),
        train_ids=list(map(str, split.train_ids)),
        train_y=list(map(int, split.train_y)),
        test_ids=list(map(str, split.test_ids)),
        test_y=list(map(int, split.test_y)),
        test_prob=y_prob_test,
        metrics_df=mdf,
        roc_img_path=str(outdir / ROC_PNG),
        pr_img_path=str(outdir / PR_PNG),
        notes=notes,
    )

    print("\n=== DONE ===")
    print(f"Outputs written to: {outdir}")
    print(f"Summary: {outdir / SUMMARY_PDF}")



if __name__ == "__main__":
    main()

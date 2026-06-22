# experiment_runner.py
"""
Core experiment loop.

For each (task, input_config, split_mode, use_bayes_opt, seed) combination:
  1.  Build train / test DataFrames.
  2.  Optionally run Bayesian hyper-parameter optimisation on train.
  3.  Fit pipeline on train, predict on test.
  4.  Record metrics.

Parallelism controls
--------------------
parallel_cv   : passes n_jobs=-1 to cross_val_score inside each BO trial.
parallel_reps : runs the NUM_REPETITIONS seeds in parallel via joblib.
                Safe with both skopt (gp_minimize replaced by Optuna) and
                default-param runs. Uses PARALLEL_REPETITIONS_N_JOBS from
                constants.py as the worker count.
"""

from __future__ import annotations

import os
import traceback
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit

from constants import (
    BASE_SEED,
    NUM_REPETITIONS,
    TRAINING_SUBJECTS,
    TESTING_SUBJECTS,
    RESPONDER_LABELS,
    BAYES_OPT_N_JOBS,
    PARALLEL_REPETITIONS_N_JOBS,
)
from data_loader import build_jmap_dataframe, build_subject_demographics
from metrics_recorder import (
    compute_clf_metrics,
    compute_reg_metrics,
    summarise_clf_metrics,
    summarise_reg_metrics,
)
from pipeline_factory import (
    attach_demo_features,
    build_clf_pipeline,
    build_reg_pipeline,
)


# ─────────────────────────────────────────────────────────────────────────────
# Split helpers
# ─────────────────────────────────────────────────────────────────────────────

def _cross_site_split(
    jmap_df: pd.DataFrame,
    demo_df: pd.DataFrame,
    task: str,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Fixed cross-site split: site 1&2 → train, site 3 → test."""
    train_ids = [s for s in TRAINING_SUBJECTS if s in jmap_df.index]
    test_ids  = [s for s in TESTING_SUBJECTS  if s in jmap_df.index]

    X_train = jmap_df.loc[train_ids]
    X_test  = jmap_df.loc[test_ids]

    if task == "classification":
        y_train = pd.Series(
            [RESPONDER_LABELS[s] for s in train_ids], index=train_ids, name="label"
        )
        y_test = pd.Series(
            [RESPONDER_LABELS[s] for s in test_ids], index=test_ids, name="label"
        )
    else:
        y_train = demo_df.loc[train_ids, "stai_state_decrease"]
        y_test  = demo_df.loc[test_ids,  "stai_state_decrease"]
        valid_train = y_train.dropna().index
        valid_test  = y_test.dropna().index
        X_train, y_train = X_train.loc[valid_train], y_train.loc[valid_train]
        X_test,  y_test  = X_test.loc[valid_test],  y_test.loc[valid_test]

    return X_train, y_train, X_test, y_test


def _mix_site_split(
    jmap_df: pd.DataFrame,
    demo_df: pd.DataFrame,
    task: str,
    random_state: int,
    test_size: float = 0.3,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Stratified random split over pooled data."""
    all_ids = list(jmap_df.index)

    if task == "classification":
        y_all = pd.Series(
            [RESPONDER_LABELS[s] for s in all_ids], index=all_ids, name="label"
        )
        splitter = StratifiedShuffleSplit(
            n_splits=1, test_size=test_size, random_state=random_state
        )
        train_idx, test_idx = next(splitter.split(all_ids, y_all))
    else:
        y_all = demo_df.loc[all_ids, "stai_state_decrease"]
        valid  = y_all.dropna().index.tolist()
        jmap_df = jmap_df.loc[valid]
        y_all   = y_all.loc[valid]
        all_ids = list(jmap_df.index)
        splitter = ShuffleSplit(
            n_splits=1, test_size=test_size, random_state=random_state
        )
        train_idx, test_idx = next(splitter.split(all_ids))

    train_ids = [all_ids[i] for i in train_idx]
    test_ids  = [all_ids[i] for i in test_idx]

    X_train, y_train = jmap_df.loc[train_ids], y_all.loc[train_ids]
    X_test,  y_test  = jmap_df.loc[test_ids],  y_all.loc[test_ids]
    return X_train, y_train, X_test, y_test


# ─────────────────────────────────────────────────────────────────────────────
# Single-repetition runner
# ─────────────────────────────────────────────────────────────────────────────

def run_one_repetition(
    task: str,
    input_config: str,
    split_mode: str,
    jmap_df: pd.DataFrame,
    demo_df: pd.DataFrame,
    seed: int,
    use_bayes_opt: bool = False,
    parallel_cv: bool = False,
) -> Optional[Dict]:
    """
    Run a single train/test repetition.

    Returns a dict of metrics, or None if the fold is not viable.
    """
    try:
        # ── 1. Split ──────────────────────────────────────────────────────
        if split_mode == "cross_site":
            X_train, y_train, X_test, y_test = _cross_site_split(
                jmap_df, demo_df, task
            )
        else:
            X_train, y_train, X_test, y_test = _mix_site_split(
                jmap_df, demo_df, task, random_state=seed
            )

        if len(X_train) == 0 or len(X_test) == 0:
            print(f"[SKIP] seed={seed}: empty split.")
            return None

        # ── 2. Attach demo features when needed ───────────────────────────
        X_train = attach_demo_features(X_train, demo_df, input_config)
        X_test  = attach_demo_features(X_test,  demo_df, input_config)

        # ── 3. Build pipeline ─────────────────────────────────────────────
        if task == "classification":
            pipeline = build_clf_pipeline(input_config, random_state=seed)
        else:
            pipeline = build_reg_pipeline(input_config, random_state=seed)

        # ── 4. Optional Bayesian optimisation ─────────────────────────────
        if use_bayes_opt:
            from bayes_opt_helper import run_bayes_opt
            _n_jobs = -1 if parallel_cv else BAYES_OPT_N_JOBS
            pipeline, best_params = run_bayes_opt(
                pipeline, X_train, y_train,
                task=task,
                random_state=seed,
                n_jobs=_n_jobs,
            )
            print(f"  [BO] best_params seed={seed}: {best_params}")

        # ── 5. Fit ────────────────────────────────────────────────────────
        pipeline.fit(X_train, y_train)

        # ── 6. Predict & metrics ──────────────────────────────────────────
        if task == "classification":
            y_pred = pipeline.predict(X_test)
            try:
                y_prob = pipeline.predict_proba(X_test)[:, 1]
            except AttributeError:
                y_prob = y_pred.astype(float)
            metrics = compute_clf_metrics(y_test.values, y_pred, y_prob)
        else:
            y_pred  = pipeline.predict(X_test)
            metrics = compute_reg_metrics(y_test.values, y_pred)

        metrics["seed"] = seed
        return metrics

    except Exception:
        print(f"[ERROR] seed={seed}, task={task}, config={input_config}, "
              f"split={split_mode}")
        traceback.print_exc()
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Full experiment
# ─────────────────────────────────────────────────────────────────────────────

def run_experiment(
    task: str,
    input_config: str,
    split_mode: str,
    jmap_df: pd.DataFrame,
    demo_df: pd.DataFrame,
    output_dir: str,
    use_bayes_opt: bool = False,
    parallel_cv: bool = False,
    parallel_reps: bool = False,
) -> pd.DataFrame:
    """
    Run NUM_REPETITIONS repetitions (seeds BASE_SEED … BASE_SEED+19)
    and save per-repetition and summary CSVs.

    Parameters
    ----------
    task          : "classification" or "regression"
    input_config  : one of the 5 input configurations
    split_mode    : "cross_site" or "mix_site"
    jmap_df       : DataFrame with jmap volume columns indexed by subject_id
    demo_df       : Subject-level demographic / STAI DataFrame
    output_dir    : folder to write results
    use_bayes_opt : whether to run BO hyper-parameter search
    parallel_cv   : parallelise CV folds inside each BO trial
    parallel_reps : parallelise the NUM_REPETITIONS seeds via joblib
                    (uses PARALLEL_REPETITIONS_N_JOBS from constants.py)

    Returns
    -------
    summary_df : DataFrame with Mean ± SD for each metric
    """
    os.makedirs(output_dir, exist_ok=True)
    bo_tag = "bayes" if use_bayes_opt else "default"
    prefix = f"{task}_{input_config}_{split_mode}_{bo_tag}"

    # ── Resolve effective repetition-level n_jobs ─────────────────────────
    # Optuna (fork-safe) replaced gp_minimize, so parallel_reps is now
    # unconditionally safe regardless of use_bayes_opt.
    effective_rep_jobs = PARALLEL_REPETITIONS_N_JOBS if parallel_reps else 1

    if parallel_reps:
        print(
            f"[INFO] Parallel repetitions enabled  "
            f"(n_jobs={effective_rep_jobs}).  "
            f"Note: each worker holds its own copy of NIfTI volumes in RAM."
        )

    seeds = [BASE_SEED + i for i in range(NUM_REPETITIONS)]

    # ── Run repetitions ───────────────────────────────────────────────────
    if effective_rep_jobs == 1:
        # Sequential path — familiar per-rep progress prints are preserved.
        all_metrics: List[Dict] = []
        for rep_idx, seed in enumerate(seeds):
            print(
                f"[{prefix}] repetition {rep_idx + 1}/{NUM_REPETITIONS}  seed={seed}"
            )
            metrics = run_one_repetition(
                task=task,
                input_config=input_config,
                split_mode=split_mode,
                jmap_df=jmap_df,
                demo_df=demo_df,
                seed=seed,
                use_bayes_opt=use_bayes_opt,
                parallel_cv=parallel_cv,
            )
            if metrics is not None:
                all_metrics.append(metrics)

    else:
        # Parallel path — joblib dispatches all seeds simultaneously.
        # stdout from workers may interleave; that is expected behaviour.
        print(
            f"[{prefix}] launching {NUM_REPETITIONS} repetitions "
            f"in parallel (n_jobs={effective_rep_jobs}) …"
        )
        results = Parallel(n_jobs=effective_rep_jobs)(
            delayed(run_one_repetition)(
                task=task,
                input_config=input_config,
                split_mode=split_mode,
                jmap_df=jmap_df,
                demo_df=demo_df,
                seed=seed,
                use_bayes_opt=use_bayes_opt,
                parallel_cv=parallel_cv,
            )
            for seed in seeds
        )
        all_metrics = [m for m in results if m is not None]

    if not all_metrics:
        print(f"[WARN] No valid repetitions for {prefix}.")
        return pd.DataFrame()

    # ── Per-repetition CSV ────────────────────────────────────────────────
    per_rep_df = pd.DataFrame(all_metrics)
    per_rep_df.to_csv(
        os.path.join(output_dir, f"{prefix}_per_repetition.csv"), index=False
    )

    # ── Summary CSV ───────────────────────────────────────────────────────
    if task == "classification":
        summary_df = summarise_clf_metrics(all_metrics)
    else:
        summary_df = summarise_reg_metrics(all_metrics)

    summary_df.insert(0, "Task",         task)
    summary_df.insert(1, "Input Config", input_config)
    summary_df.insert(2, "Split Mode",   split_mode)
    summary_df.insert(3, "BO",           bo_tag)
    summary_df.to_csv(
        os.path.join(output_dir, f"{prefix}_summary.csv"), index=False
    )

    print(f"\n{'─'*60}")
    print(f"  Summary: {prefix}")
    print(summary_df.to_string(index=False))
    print(f"{'─'*60}\n")

    return summary_df
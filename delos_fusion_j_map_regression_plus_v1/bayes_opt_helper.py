# bayes_opt_helper.py
"""
Bayesian hyper-parameter optimisation using Optuna (replaces skopt).
Optuna uses a fork-safe in-memory study, so parallel_reps works correctly.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Tuple

import numpy as np
import optuna
from imblearn.pipeline import Pipeline
from sklearn.base import clone as sk_clone
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score

from constants import BAYES_OPT_CV_FOLDS, BAYES_OPT_N_ITER, BAYES_OPT_N_JOBS

optuna.logging.set_verbosity(optuna.logging.WARNING)  # suppress optuna noise


def _format_params(params: dict) -> str:
    parts = []
    for k, v in params.items():
        short_key = k.split("__")[-1]
        if isinstance(v, float):
            parts.append(f"{short_key}={v:.4g}")
        else:
            parts.append(f"{short_key}={v}")
    return "  ".join(parts)


def _progress_bar(current: int, total: int, width: int = 30) -> str:
    filled = int(width * current / total)
    bar    = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {current}/{total}"


def run_bayes_opt(
    pipeline: Pipeline,
    X_train: Any,
    y_train: Any,
    task: str = "classification",
    random_state: int = 42,
    n_jobs: int = BAYES_OPT_N_JOBS,
) -> Tuple[Pipeline, Dict]:

    if task == "classification":
        cv           = StratifiedKFold(
            n_splits=BAYES_OPT_CV_FOLDS, shuffle=True, random_state=random_state
        )
        scorer       = "roc_auc"
        scorer_label = "AUROC"
    else:
        cv           = KFold(
            n_splits=BAYES_OPT_CV_FOLDS, shuffle=True, random_state=random_state
        )
        scorer       = "neg_mean_absolute_error"
        scorer_label = "MAE"

    state = {
        "call":        0,
        "best_score":  -np.inf,
        "best_params": None,
        "start_time":  time.time(),
        "trial_scores": [],
    }

    sep = "─" * 68
    print(f"\n{sep}")
    print(f"  Bayesian Optimisation (Optuna)  |  task={task}  |  "
          f"{BAYES_OPT_N_ITER} iterations")
    print(f"  Inner CV: {BAYES_OPT_CV_FOLDS}-fold  |  scorer: {scorer_label}")
    print(sep)

    def objective(trial: optuna.Trial) -> float:
        state["call"] += 1
        call_idx = state["call"]
        t0       = time.time()

        # ── Suggest hyperparameters ───────────────────────────────────────
        params = {
            "rbf__gamma":        trial.suggest_float(
                "rbf__gamma", 1e-3, 1e1, log=True
            ),
            "rbf__n_components": trial.suggest_int(
                "rbf__n_components", 10, 200
            ),
        }

        pipe = sk_clone(pipeline)
        pipe.set_params(**params)

        try:
            scores     = cross_val_score(
                pipe, X_train, y_train,
                cv=cv,
                scoring=scorer,
                n_jobs=n_jobs,
                error_score=-np.inf,
            )
            mean_score = float(np.nanmean(scores))
            std_score  = float(np.nanstd(scores))
            if not np.isfinite(mean_score):
                mean_score = -np.inf
        except Exception as exc:
            print(f"    [trial {call_idx:03d}] ERROR: {exc}")
            mean_score = -np.inf
            std_score  = 0.0

        is_best = mean_score > state["best_score"]
        if is_best:
            state["best_score"]  = mean_score
            state["best_params"] = params.copy()

        state["trial_scores"].append(mean_score)

        elapsed_trial = time.time() - t0
        elapsed_total = time.time() - state["start_time"]
        avg_trial_time  = elapsed_total / call_idx
        eta_seconds     = avg_trial_time * (BAYES_OPT_N_ITER - call_idx)

        if scorer == "neg_mean_absolute_error":
            score_str = f"{scorer_label}={-mean_score:.4f} ± {std_score:.4f}"
            best_str  = f"best {scorer_label}={-state['best_score']:.4f}"
        else:
            score_str = f"{scorer_label}={mean_score:.4f} ± {std_score:.4f}"
            best_str  = f"best {scorer_label}={state['best_score']:.4f}"

        marker  = " ◄ NEW BEST" if is_best else ""
        eta_str = (
            f"{int(eta_seconds // 60)}m {int(eta_seconds % 60)}s"
            if eta_seconds > 0 else "─"
        )
        print(
            f"  {_progress_bar(call_idx, BAYES_OPT_N_ITER)}  trial {call_idx:03d}"
            f"  {score_str}"
            f"  ({elapsed_trial:.1f}s/trial  ETA {eta_str})"
            f"{marker}"
        )
        print(f"    params: {_format_params(params)}")
        if is_best:
            print(f"    ↳ {best_str}")

        # Optuna minimises by default
        return -mean_score if np.isfinite(mean_score) else 0.0

    # ── Run optimisation ──────────────────────────────────────────────────
    sampler = optuna.samplers.TPESampler(seed=random_state)
    study   = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=BAYES_OPT_N_ITER)

    total_time  = time.time() - state["start_time"]
    best_params = state["best_params"]

    valid_scores = [s for s in state["trial_scores"] if np.isfinite(s)]
    if scorer == "neg_mean_absolute_error":
        score_summary = (
            f"  best {scorer_label} : {-state['best_score']:.4f}\n"
            f"  worst {scorer_label}: {-min(valid_scores):.4f}\n"
            f"  mean  {scorer_label}: {-np.mean(valid_scores):.4f}"
        )
    else:
        score_summary = (
            f"  best  {scorer_label}: {state['best_score']:.4f}\n"
            f"  worst {scorer_label}: {min(valid_scores):.4f}\n"
            f"  mean  {scorer_label}: {np.mean(valid_scores):.4f}"
        )

    print(f"\n{sep}")
    print(f"  BO complete  |  total time: "
          f"{int(total_time // 60)}m {int(total_time % 60)}s")
    print(score_summary)
    print(f"  best params: {_format_params(best_params)}")
    print(sep)

    pipeline.set_params(**best_params)
    return pipeline, best_params
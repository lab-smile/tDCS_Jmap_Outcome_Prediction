"""
Training and evaluation entry points.

Key responsibilities:
- Build the sklearn Pipeline (our custom JmapACTPreprocessor etc.)
- Hyperparameter tuning on training set with repeated 3-fold CV
- Fit best model on full training set
- Predict probabilities on test set
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, List
import time
import threading
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from imblearn.pipeline import Pipeline
from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
)

def compute_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "AUROC": roc_auc_score(y_true, y_prob),
        "AUPRC": average_precision_score(y_true, y_prob),
        "BalancedAccuracy": balanced_accuracy_score(y_true, y_pred),
        "WeightedF1": f1_score(y_true, y_pred, average="weighted"),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


def compute_rep_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)

    return {
        "AUROC": roc_auc_score(y_true, y_prob),
        "AUPRC": average_precision_score(y_true, y_prob),
        "BalancedAccuracy": balanced_accuracy_score(y_true, y_pred),
        "WeightedF1": f1_score(y_true, y_pred, average="weighted"),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }

def add_ci_from_repetitions(rep_df, alpha=0.05):
    rows = []
    for metric in ["AUROC", "AUPRC", "BalancedAccuracy", "WeightedF1", "MCC"]:
        lo = rep_df[metric].quantile(alpha / 2)
        hi = rep_df[metric].quantile(1 - alpha / 2)
        mean = rep_df[metric].mean()
        std = rep_df[metric].std(ddof=1)

        rows.append({
            "Metric": metric,
            "Mean ± SD": f"{mean:.3f} ± {std:.3f}",
            "95% CI (Lower – Upper)": f"{lo:.3f} – {hi:.3f}",
            "Mean": mean,
            "SD": std,
            "95% CI Lower": lo,
            "95% CI Upper": hi,
        })
    return pd.DataFrame(rows)



# Optional dependencies for advanced search
try:
    from skopt import BayesSearchCV
except Exception:
    BayesSearchCV = None

try:
    from sklearn_genetic import GASearchCV
    from sklearn_genetic.space import Continuous, Integer
except Exception:
    GASearchCV = None
    Continuous = Integer = None

@dataclass
class ProgressState:
    stage: str = "init"
    done: int = 0
    total: int = 0
    start_time: float = 0.0

class MinuteReporter:
    def __init__(self, state: ProgressState, every_sec: int = 60):
        self.state = state
        self.every_sec = every_sec
        self._stop = threading.Event()
        self._thr: Optional[threading.Thread] = None

    def start(self):
        self.state.start_time = time.time()
        self._thr = threading.Thread(target=self._run, daemon=True)
        self._thr.start()

    def stop(self):
        self._stop.set()
        if self._thr is not None:
            self._thr.join(timeout=1)

    def _run(self):
        last = 0.0
        while not self._stop.is_set():
            time.sleep(1)
            now = time.time()
            if now - last < self.every_sec:
                continue
            last = now
            done = max(0, self.state.done)
            total = max(0, self.state.total)
            elapsed = now - self.state.start_time
            eta = None
            if total > 0 and done > 0:
                rate = elapsed / done
                eta = rate * (total - done)
            if eta is None:
                print(f"[MinuteReporter progress] {self.state.stage} | done={done}/{total} | elapsed={elapsed/60:.1f} min")
            else:
                print(f"[MinuteReporter progress] {self.state.stage} | done={done}/{total} | elapsed={elapsed/60:.1f} min | ETA={eta/60:.1f} min")

@dataclass
class TrainResult:
    best_estimator: Any
    best_params: Dict[str, Any]
    cv_best_score: float

def repeated_nested_cv_oof(
    X: Any,
    y: np.ndarray,
    strategy: str,
    random_state: int,
    outer_n_splits: int,
    inner_n_splits: int,
    inner_n_repeats: int,
    n_reps: int,
    report_every_sec: int = 60,
):
    """
    BOOTSTRAP_N repetitions of OUTER_N_SPLITS-fold nested stratified CV.

    Returns
    -------
    rep_metrics : List[Dict[str, float]]
        One dict per repetition with AUROC/AUPRC/etc computed on that rep's OOF predictions.
    oof_prob_mean : np.ndarray
        Mean predicted probability per subject across repetitions (each rep supplies one OOF pred per subject).
    diagnostics : Dict[str, Any]
        Includes per-rep outer fold AUCs, params, etc.
    """

    y = np.asarray(y, dtype=int)
    n = len(y)

    # accumulate OOF probabilities across reps to get stable per-subject mean
    oof_sum = np.zeros(n, dtype=float)
    oof_cnt = np.zeros(n, dtype=int)

    rep_metrics = []        # one row per repetition (OOF pooled)
    fold_metrics = []       # one row per outer fold per repetition
    rep_outer_auc_scores = []
    rep_outer_best_params = []

    state = ProgressState(stage="mixed-site repeated nested CV", done=0, total=n_reps)
    reporter = MinuteReporter(state, every_sec=report_every_sec)
    reporter.start()

    try:
        for rep in range(n_reps):
            rs = int(random_state) + rep  # deterministic, different each repetition

            outer = StratifiedKFold(
                n_splits=outer_n_splits,
                shuffle=True,
                random_state=rs,
            )

            oof_prob = np.full(n, np.nan, dtype=float)
            outer_scores = []
            outer_best_params = []

            state.stage = f"rep {rep+1}/{n_reps}: outer CV with inner tuning"

            for fold_i, (tr_idx, va_idx) in enumerate(outer.split(np.zeros_like(y), y), start=1):
                X_tr = X.iloc[tr_idx] if hasattr(X, "iloc") else X[tr_idx]
                y_tr = y[tr_idx]
                X_va = X.iloc[va_idx] if hasattr(X, "iloc") else X[va_idx]
                y_va = y[va_idx]

                tr_res = tune_model(
                    X_train=X_tr,
                    y_train=y_tr,
                    strategy=strategy,
                    random_state=rs,
                    n_splits=inner_n_splits,
                    n_repeats=inner_n_repeats,
                )

                y_va_prob = predict_proba_positive(tr_res.best_estimator, X_va)
                oof_prob[va_idx] = y_va_prob

                # -------- compute fold-level metrics on this outer test fold --------
                fm = compute_metrics(y_true=y_va, y_prob=y_va_prob, threshold=0.5)
                fm.update({
                    "rep": rep + 1,
                    "fold": fold_i,
                    "n_fold": int(len(va_idx)),
                    "pos_fold": int(np.sum(y_va)),
                    "neg_fold": int(len(va_idx) - np.sum(y_va)),
                })
                fold_metrics.append(fm)
                # -----------------------------------------------------------------------

                auc = roc_auc_score(y_va, y_va_prob)
                outer_scores.append(float(auc))
                outer_best_params.append(dict(tr_res.best_params))
            
            # repetition-level (pooled OOF over all folds in that repetition)
            rm = compute_metrics(y_true=y, y_prob=oof_prob, threshold=0.5)
            rm["rep"] = rep + 1
            rep_metrics.append(rm)

            # metrics for this repetition (computed on OOF preds)
            auroc = float(roc_auc_score(y, oof_prob))
            auprc = float(average_precision_score(y, oof_prob))

            rep_metric = compute_rep_metrics(y, oof_prob, threshold=0.5)
            rep_metric["rep"] = rep + 1
            rep_metrics.append(rep_metric)
            
            rep_outer_auc_scores.append(outer_scores)
            rep_outer_best_params.append(outer_best_params)

            # accumulate mean OOF prob per subject
            valid = np.isfinite(oof_prob)
            oof_sum[valid] += oof_prob[valid]
            oof_cnt[valid] += 1

            state.done += 1

    finally:
        reporter.stop()

    oof_prob_mean = oof_sum / np.maximum(oof_cnt, 1)

    diagnostics = {
        "rep_outer_auc_scores": rep_outer_auc_scores,
        "rep_outer_best_params": rep_outer_best_params,
        "oof_count_min": int(oof_cnt.min()),
        "oof_count_max": int(oof_cnt.max()),
    }
    return rep_metrics, fold_metrics, oof_prob_mean, diagnostics


def build_pipeline(random_state: int = 42) -> Pipeline:
    """
    Import your project-specific pipeline components.
    If these imports fail, you'll get a clear error message.
    """
    try:
        from .ml_clinical_act_jmap.jmap_preprocessor import JmapACTPreprocessor
        from .ml_clinical_act_jmap.pca_with_names import PCAWithNames
        from .ml_clinical_act_jmap.safe_smote import SafeSMOTE
        from .ml_clinical_act_jmap.hetero_selector import WelchTTestSelector, MRMRSelector
    except Exception as e:
        raise ImportError(
            "Could not import your custom ML components. "
            "Ensure your PYTHONPATH includes the package that contains ml_clinical_act_jmap.\n"
            f"Original error: {e}"
        )

    # sklearn components
    from sklearn.kernel_approximation import RBFSampler
    from sklearn.linear_model import SGDClassifier

    pipeline = Pipeline([
        ("prep", JmapACTPreprocessor(
            jmap_features=["jmap_tp1"],
            strategy="flatten",          # "stats", "flatten", or "pca"
            n_components=8,              # used only if strategy="pca"
            keep_channel_axis=True,      # if data are 4D (X,Y,Z,C)
            random_state=random_state,
            atlas_path="../hammers_atlas/Hammers_mith_atlas_n30r83_SPM5.nii.gz",
            atlas_labels_path="../hammers_atlas/n30r83_id2name_clean.txt",
            scale_volume=True
        )),
        ("smote", SafeSMOTE(sampling_strategy=1.0, k_neighbors=2, random_state=random_state)),
        ("pca", PCAWithNames(n_components=0.95, random_state=random_state)),
        ("ttest", WelchTTestSelector(p_thresh=1e-4, min_k_if_empty=2000, cap_after_t=15000)),
        ("mrmr",  MRMRSelector(frac_for_topk=0.01, min_topk=10, max_topk=20)),
        ("rbf",   RBFSampler(gamma=1.0, n_components=300, random_state=random_state)),
        ("sgd",   SGDClassifier(loss="log_loss", max_iter=1000, random_state=random_state)),
    ])
    return pipeline


def hyperparam_space() -> Dict[str, Any]:
    """
    Hyperparameters to search:
    - mRMR top features retained: 10 - 500
    - RBF gamma: 1e-3 - 1e1 (log)
    - number of random Fourier features: 100 - 1000
    """
    return {
        "mrmr__max_topk": list(range(10, 501, 10)),
        "rbf__gamma": np.logspace(-3, 1, 20),
        "rbf__n_components": list(range(100, 1001, 100)),
    }


def tune_model(
    X_train: Any,
    y_train: np.ndarray,
    strategy: str = "random",
    random_state: int = 42,
    n_splits: int = 3,
    n_repeats: int = 20,
    n_iter_random: int = 1,
) -> TrainResult:
    """
    Uses repeated stratified k-fold CV within training set.
    Scoring: ROC AUC (uses predict_proba / decision_function)
    """
    pipe = build_pipeline(random_state=random_state)

    cv = RepeatedStratifiedKFold(
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=random_state,
    )

    params = hyperparam_space()
    scoring = "roc_auc"

    strategy = strategy.lower().strip()

    if strategy == "grid":
        search = GridSearchCV(
            pipe, params, scoring=scoring, cv=cv, n_jobs=-1, refit=True, verbose=2
        )
    elif strategy == "random":
        search = RandomizedSearchCV(
            pipe, params, n_iter=n_iter_random, scoring=scoring, cv=cv, n_jobs=-1,
            refit=True, verbose=2, random_state=random_state
        )
    elif strategy == "bayes":
        if BayesSearchCV is None:
            raise ImportError("BayesSearchCV not available. Install scikit-optimize: pip install scikit-optimize")
        # Build Bayes spaces
        bayes_spaces = {
            "mrmr__max_topk": Integer(10, 500),
            "rbf__gamma": Continuous(1e-3, 1e1, prior="log-uniform"),
            "rbf__n_components": Integer(100, 1000),
        }
        search = BayesSearchCV(
            pipe, bayes_spaces, n_iter=n_iter_random, scoring=scoring, cv=cv,
            n_jobs=-1, refit=True, verbose=2, random_state=random_state
        )
    elif strategy == "genetic":
        if GASearchCV is None:
            raise ImportError("GASearchCV not available. Install sklearn-genetic-opt: pip install sklearn-genetic-opt")
        ga_spaces = {
            "mrmr__max_topk": Integer(10, 500),
            "rbf__gamma": Continuous(1e-3, 1e1, prior="log-uniform"),
            "rbf__n_components": Integer(100, 1000),
        }
        search = GASearchCV(
            estimator=pipe,
            cv=cv,
            scoring=scoring,
            population_size=20,
            generations=20,
            n_jobs=-1,
            verbose=True,
            keep_top_k=4,
            crossover_probability=0.7,
            mutation_probability=0.2,
            param_grid=ga_spaces,
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    search.fit(X_train, y_train)

    return TrainResult(
        best_estimator=search.best_estimator_,
        best_params=dict(search.best_params_),
        cv_best_score=float(search.best_score_),
    )


def predict_proba_positive(model: Any, X: Any) -> np.ndarray:
    """
    Return P(y=1) for each sample.
    """
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(X)
        return p[:, 1]
    # Fall back to decision_function with logistic transform
    if hasattr(model, "decision_function"):
        s = model.decision_function(X)
        return 1.0 / (1.0 + np.exp(-s))
    raise AttributeError("Model has neither predict_proba nor decision_function")


def fit_best_and_predict(
    X_train: Any,
    y_train: np.ndarray,
    X_test: Any,
    strategy: str,
    random_state: int,
    n_splits: int,
    n_repeats: int,
) -> Tuple[TrainResult, np.ndarray]:
    train_res = tune_model(
        X_train=X_train,
        y_train=y_train,
        strategy=strategy,
        random_state=random_state,
        n_splits=n_splits,
        n_repeats=n_repeats,
    )
    y_prob_test = predict_proba_positive(train_res.best_estimator, X_test)
    return train_res, y_prob_test

def nested_outer_cv_fit_and_predict(
    X_train: Any,
    y_train: np.ndarray,
    X_test: Any,
    strategy: str,
    random_state: int,
    outer_n_splits: int,
    inner_n_splits: int,
    inner_n_repeats: int,
    report_every_sec: int = 60,
) -> Tuple[TrainResult, np.ndarray, Dict[str, Any]]:
    """
    Nested CV:
      - Outer CV on training split only (to estimate performance on train split)
      - Inner repeated CV for hyperparameter tuning in each outer fold
      - Final tune on full training split and predict on held-out test

    Returns:
      final TrainResult (fit on full training),
      y_prob_test,
      diagnostics dict with outer fold scores and summary
    """
    y_train = np.asarray(y_train, dtype=int)

    outer = StratifiedKFold(
        n_splits=outer_n_splits,
        shuffle=True,
        random_state=random_state,
    )

    state = ProgressState(stage="outer-cv", done=0, total=outer_n_splits + 1)  # +1 final fit
    reporter = MinuteReporter(state, every_sec=report_every_sec)
    reporter.start()

    outer_scores = []
    outer_best_params = []

    try:
        for fold_i, (tr_idx, va_idx) in enumerate(outer.split(np.zeros_like(y_train), y_train), start=1):
            state.stage = f"outer-cv fold {fold_i}/{outer_n_splits} (inner tuning)"
            X_tr = X_train.iloc[tr_idx] if hasattr(X_train, "iloc") else X_train[tr_idx]
            y_tr = y_train[tr_idx]
            X_va = X_train.iloc[va_idx] if hasattr(X_train, "iloc") else X_train[va_idx]
            y_va = y_train[va_idx]

            # inner tuning on outer-train fold
            tr_res = tune_model(
                X_train=X_tr,
                y_train=y_tr,
                strategy=strategy,
                random_state=random_state,
                n_splits=inner_n_splits,
                n_repeats=inner_n_repeats,
            )

            # evaluate on outer-val fold
            y_va_prob = predict_proba_positive(tr_res.best_estimator, X_va)
            auc = roc_auc_score(y_va, y_va_prob)
            outer_scores.append(float(auc))
            outer_best_params.append(dict(tr_res.best_params))

            state.done += 1

        # final fit on all training data (with inner tuning)
        state.stage = "final fit on full training (inner tuning)"
        final_res = tune_model(
            X_train=X_train,
            y_train=y_train,
            strategy=strategy,
            random_state=random_state,
            n_splits=inner_n_splits,
            n_repeats=inner_n_repeats,
        )
        y_prob_test = predict_proba_positive(final_res.best_estimator, X_test)
        state.done += 1

    finally:
        reporter.stop()

    diagnostics = {
        "outer_auc_scores": outer_scores,
        "outer_auc_mean": float(np.mean(outer_scores)) if len(outer_scores) else float("nan"),
        "outer_auc_std": float(np.std(outer_scores)) if len(outer_scores) else float("nan"),
        "outer_best_params": outer_best_params,
    }
    return final_res, y_prob_test, diagnostics
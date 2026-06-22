# mpi_search.py
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from mpi4py import MPI
import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, ParameterGrid
from sklearn.base import clone
from sklearn.metrics import get_scorer
from threadpoolctl import threadpool_limits

# --- add near the top of mpi_search.py ---
def _row_slice(X, idx):
    """Row-wise slice that works for numpy arrays and pandas DataFrames/Series."""
    try:
        # pandas DataFrame/Series path
        return X.iloc[idx]
    except AttributeError:
        # numpy array / sparse matrix path
        return X[idx]

def _try_set(est, **kwargs):
    """Best-effort set_params (e.g., to pin inner threads)."""
    try:
        est.set_params(**kwargs)
    except Exception:
        pass

def _seed_estimator(estimator, base_seed, split_idx):
    """
    Derive a deterministic seed per split to vary stochastic parts (PCA, boosters, etc.)
    without leaking between candidates. Adjust key as needed (e.g., 'clf__random_state').
    """
    seed = int((base_seed + 9973 * split_idx) % (2**31 - 1))
    try:
        estimator.set_params(**{
            "clf__random_state": seed,
            "mrmr__random_state": seed,
            # add other steps if they expose `random_state`
        })
    except Exception:
        pass
    return estimator



def train_by_grid_search_cv_mpi(self, X, y, param_grid, scoring=None,
                                random_state=42, n_splits=3, n_repeats=100,
                                repeated=True, verbose=True):
    """
    MPI-distributed grid search where *repeats/folds* are split across ranks.
    - Total evaluations = n_splits * (n_repeats if repeated else 1)
    - Rank r only evaluates splits whose global index i satisfies (i % size == r)
    - We aggregate (sum, sumsq, count) per param across ranks → mean/std
    - Best params are broadcast; every rank refits on full data and returns a fitted pipeline.

    Returns:
        (fitted_best_pipeline, None, None)
    """
    # ==== MPI setup ====
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # keep inner math single-threaded on every rank
    threadpool_limits(limits=1)

    # ==== Build CV splits ====
    if repeated:
        splitter = RepeatedStratifiedKFold(
            n_splits=n_splits, n_repeats=n_repeats, random_state=random_state
        )
    else:
        splitter = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=random_state
        )

    all_splits = list(splitter.split(X, y))
    total_splits = len(all_splits)
    # Partition splits across ranks by index modulo size
    my_split_indices = [i for i in range(total_splits) if i % size == rank]

    if rank == 0 and verbose:
        print(f"[MPI] ranks={size} | candidates={len(list(ParameterGrid(param_grid)))} | "
              f"splits/candidate={total_splits} "
              f"(~{len(my_split_indices)} per rank)")
        for r in range(size):
            count_r = sum(1 for i in range(total_splits) if i % size == r)
            print(f"[MPI] rank {r}: {count_r} splits")

    # ==== Scorer ====
    scorer = get_scorer(scoring) if isinstance(scoring, str) else scoring

    # ==== Evaluate each candidate, only on this rank's splits ====
    grid = list(ParameterGrid(param_grid))

    # For each candidate, we compute partial sums locally
    # results_local: list of tuples (idx, sum, sumsq, n, params)
    results_local = []

    for gi, params in enumerate(grid):
        est_template = clone(self.pipeline)
        est_template.set_params(**params)

        # pin inner threads for common libs (e.g., xgboost's nthread)
        _try_set(est_template, **{"clf__nthread": 1})

        # partial accumulators
        sum_scores = 0.0
        sumsq_scores = 0.0
        n_evals = 0

        for local_k, split_idx in enumerate(my_split_indices):
            tr, te = all_splits[split_idx]
            fold_est = clone(est_template)

            # give a deterministic but varying seed per split
            fold_est = _seed_estimator(fold_est, base_seed=random_state, split_idx=split_idx)

            Xtr, Xte = _row_slice(X, tr), _row_slice(X, te)
            ytr, yte = _row_slice(y, tr), _row_slice(y, te)

            fold_est.fit(Xtr, ytr)
            if scorer is not None:
                s = float(scorer(fold_est, Xte, yte))
            else:
                s = float(fold_est.score(Xte, yte))

            sum_scores += s
            sumsq_scores += s * s
            n_evals += 1

            if verbose and (local_k + 1) % max(1, len(my_split_indices)//5 or 1) == 0:
                print(f"[Rank {rank}] cand {gi+1}/{len(grid)} "
                      f"| split {local_k+1}/{len(my_split_indices)} "
                      f"| partial mean={sum_scores/max(1,n_evals):.4f}")

        results_local.append((gi, sum_scores, sumsq_scores, n_evals, params))

    # ==== Gather & aggregate on rank 0 ====
    gathered = comm.gather(results_local, root=0)

    if rank == 0:
        # Initialize global accumulators per candidate
        acc = {gi: {"sum": 0.0, "sumsq": 0.0, "n": 0, "params": None} for gi in range(len(grid))}
        for chunk in gathered:
            for gi, s, ss, n, params in chunk:
                acc[gi]["sum"] += s
                acc[gi]["sumsq"] += ss
                acc[gi]["n"] += n
                acc[gi]["params"] = params

        # Compute global mean/std and select best
        summary = []
        for gi in range(len(grid)):
            a = acc[gi]
            n = max(1, a["n"])
            mean = a["sum"] / n
            # unbiased std over aggregate: sqrt(E[x^2] - (E[x])^2)
            var = max(a["sumsq"] / n - mean * mean, 0.0)
            std = np.sqrt(var)
            summary.append((gi, mean, std, a["params"]))

        best = max(summary, key=lambda t: t[1])
        _, best_mean, best_std, best_params = best
        if verbose:
            print(f"[MPI] Best params: {best_params} | mean={best_mean:.6f} ± {best_std:.6f}")
    else:
        best_params = None

    # ==== Broadcast best params; refit on full data on every rank ====
    best_params = comm.bcast(best_params, root=0)

    best_est = clone(self.pipeline)
    best_est.set_params(**best_params)
    _try_set(best_est, **{"clf__nthread": 1})
    # final refit can use a fixed seed
    best_est = _seed_estimator(best_est, base_seed=random_state, split_idx=0)
    best_est.fit(X, y)

    self.pipeline = best_est
    return self.pipeline

# main.py
"""
Entry point for all SMILE ACT tDCS J-Map ML experiments.

Runs:
  ┌─────────────────────────────────┬───────────────────────────────┐
  │ Input configurations            │ Split modes                   │
  ├─────────────────────────────────┼───────────────────────────────┤
  │ 1. magnitude                    │ cross_site                    │
  │ 2. theta_phi                    │ mix_site                      │
  │ 3. magnitude_theta_phi          │                               │
  │ 4. jxyz                         │                               │
  │ 5. magnitude_demo               │                               │
  └─────────────────────────────────┴───────────────────────────────┘

  × 2 tasks (classification, regression)
  × 2 BO modes (default, bayes)
  × 20 random seeds  (42 … 61)

Output is written to:
  output/
    {task}/{split_mode}/{input_config}/{bo_mode}/
      *_per_repetition.csv
      *_summary.csv
  output/
    all_summaries.csv   ← aggregated table

Parallelism flags
-----------------
  --parallel_cv    parallelise CV folds inside each BO trial
                   (overrides BAYES_OPT_N_JOBS → -1)
  --parallel_reps  parallelise the 20 repetition seeds
                   (uses PARALLEL_REPETITIONS_N_JOBS from constants;
                    auto-disabled when --bayes_only is active)
"""

import argparse
import os

import pandas as pd

from constants import (
    OUTPUT_ROOT,
    TRAINING_SUBJECTS,
    TESTING_SUBJECTS,
    PARALLEL_REPETITIONS_N_JOBS,
)
from data_loader import ActDataImport, build_jmap_dataframe, build_subject_demographics


# ─────────────────────────────────────────────────────────────────────────────
# Configuration grid
# ─────────────────────────────────────────────────────────────────────────────

INPUT_CONFIGS = [
    #"magnitude",
    # "theta_phi",
    #"magnitude_theta_phi",
    #"jxyz",
    "magnitude_demo",
]

TASKS       = ["classification", "regression"]
#TASKS       = ["regression"]
SPLIT_MODES = ["cross_site", "mix_site"]
BO_MODES    = [True]   # False = default params, True = Bayesian opt


# ─────────────────────────────────────────────────────────────────────────────
# Argument parser
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="SMILE ACT tDCS J-Map ML pipeline"
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=TASKS,
        choices=TASKS,
        help="Which tasks to run (default: both)",
    )
    parser.add_argument(
        "--input_configs",
        nargs="+",
        default=INPUT_CONFIGS,
        choices=INPUT_CONFIGS,
        help="Which input configurations to run (default: configured in script)",
    )
    parser.add_argument(
        "--split_modes",
        nargs="+",
        default=SPLIT_MODES,
        choices=SPLIT_MODES,
        help="Which split modes to run (default: both)",
    )
    parser.add_argument(
        "--no_bayes",
        action="store_true",
        help="Skip Bayesian optimisation runs (only run default params)",
    )
    parser.add_argument(
        "--bayes_only",
        action="store_true",
        help="Only run Bayesian optimisation (skip default-param runs)",
    )
    parser.add_argument(
        "--output_root",
        default=OUTPUT_ROOT,
        help=f"Root output directory (default: {OUTPUT_ROOT})",
    )
    parser.add_argument(
        "--parallel_cv",
        action="store_true",
        help=(
            "Parallelise CV folds inside each BO trial "
            "(sets n_jobs=-1 in cross_val_score, overriding BAYES_OPT_N_JOBS)."
        ),
    )
    parser.add_argument(
        "--parallel_reps",
        action="store_true",
        default=(PARALLEL_REPETITIONS_N_JOBS != 1),  # honour constant if set
        help=(
            "Parallelise the 20 repetition seeds using joblib "
            f"(n_jobs=PARALLEL_REPETITIONS_N_JOBS={PARALLEL_REPETITIONS_N_JOBS} "
            "from constants.py). "
            "Auto-disabled when Bayesian optimisation is active. "
            "Default mirrors PARALLEL_REPETITIONS_N_JOBS != 1."
        ),
    )
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # ── Validate mutually exclusive flags ─────────────────────────────────
    if args.no_bayes and args.bayes_only:
        raise ValueError("Cannot combine --no_bayes and --bayes_only.")

    # ── Resolve BO modes ──────────────────────────────────────────────────
    if args.no_bayes:
        bo_modes = [False]
    elif args.bayes_only:
        bo_modes = [True]
    else:
        bo_modes = BO_MODES

    # ── Log active parallelism settings ───────────────────────────────────
    print("─" * 65)
    print("  Parallelism settings")
    print(f"    parallel_cv   = {args.parallel_cv}"
          f"  (CV folds inside BO trials)")
    print(f"    parallel_reps = {args.parallel_reps}"
          f"  (repetition seeds, n_jobs={PARALLEL_REPETITIONS_N_JOBS})")
    if args.parallel_reps and True in bo_modes:
        print(
            "  [NOTE] parallel_reps will be auto-disabled for BO runs "
            "(gp_minimize is not fork-safe)."
        )
    print("─" * 65)

    # ── Load EHR / demographics ───────────────────────────────────────────
    print("Loading EHR data …")
    act     = ActDataImport()
    demo_df = build_subject_demographics(act.input_dataset)
    print(f"  Demographics loaded: {demo_df.shape[0]} subjects")

    # ── Pre-load all required NIfTI volumes per input config ──────────────
    all_subjects = TRAINING_SUBJECTS + TESTING_SUBJECTS
    jmap_cache   = {}
    for cfg in args.input_configs:
        print(f"Loading NIfTI volumes for config '{cfg}' …")
        df = build_jmap_dataframe(all_subjects, cfg)
        jmap_cache[cfg] = df
        print(f"  Loaded {len(df)} subjects for '{cfg}'")

    # ── Experiment loop ───────────────────────────────────────────────────
    all_summaries = []

    for task in args.tasks:
        for cfg in args.input_configs:
            for split in args.split_modes:
                for use_bo in bo_modes:
                    bo_tag  = "bayes" if use_bo else "default"
                    out_dir = os.path.join(
                        args.output_root, task, split, cfg, bo_tag
                    )
                    print(
                        f"\n{'='*65}\n"
                        f"  Task={task}  Config={cfg}  "
                        f"Split={split}  BO={bo_tag}\n"
                        f"{'='*65}"
                    )

                    from experiment_runner import run_experiment
                    summary = run_experiment(
                        task=task,
                        input_config=cfg,
                        split_mode=split,
                        jmap_df=jmap_cache[cfg],
                        demo_df=demo_df,
                        output_dir=out_dir,
                        use_bayes_opt=use_bo,
                        parallel_cv=args.parallel_cv,
                        parallel_reps=args.parallel_reps,
                    )
                    if not summary.empty:
                        all_summaries.append(summary)

    # ── Write aggregated summary ──────────────────────────────────────────
    if all_summaries:
        aggregated = pd.concat(all_summaries, ignore_index=True)
        agg_path   = os.path.join(args.output_root, "all_summaries.csv")
        os.makedirs(args.output_root, exist_ok=True)
        aggregated.to_csv(agg_path, index=False)
        print(f"\nAll summaries saved to: {agg_path}")
        print(aggregated.to_string(index=False))


if __name__ == "__main__":
    main()
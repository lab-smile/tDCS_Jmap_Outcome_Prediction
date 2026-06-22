# Contributing to HPO QuickLab Hold-Out Experiments

Thank you for your interest in contributing to this repository. This project provides reproducible hyperparameter optimization experiments using hold-out validation and SLURM-based execution.

Contributions should help improve experiment reliability, reproducibility, documentation, or code quality.

## Ways to Contribute

You can contribute by:

- Fixing bugs in experiment scripts
- Improving documentation
- Adding clearer comments to code
- Improving result logging or output formatting
- Adding reproducibility checks
- Improving SLURM job scripts
- Adding new experiment versions
- Adding tests or validation scripts
- Reporting issues with clear reproduction steps

## Before You Start

Before making changes, please check the existing issues and README files to understand the current experiment setup.

Make sure your changes are consistent with the purpose of this repository:

- Hyperparameter optimization
- Hold-out validation
- Fair comparison between search algorithms
- SLURM-based HPC execution
- Reproducible experiment results


## Code Style

Please keep code simple, readable, and reproducible.

General expectations:

* Use clear variable names
* Keep experiment parameters easy to modify
* Avoid hard-coded absolute paths
* Use random seeds where randomness is involved
* Print useful progress messages
* Save or report key results clearly
* Keep SLURM scripts portable across clusters when possible

## Experiment Reproducibility

When modifying or adding experiments, please document:

* Dataset used
* Model architecture
* Search algorithms used
* Hyperparameter search space
* Budget or stopping condition
* Random seed
* Metrics reported
* Expected output files

This helps other users understand and reproduce the results.


## Reporting Bugs

When opening an issue, include:

* Experiment folder name
* Command used to run the experiment
* SLURM job ID, if available
* Python version
* Conda environment name
* Error message or traceback
* Relevant `.out` and `.err` log content
* Steps to reproduce the issue

Example:

```text
Experiment: j-map tdcs brain direction_ml (current-density direction prediction) 
Command: sbatch run_experiment.sh
Environment: ml_exp
Python: 3.9
SLURM job ID: 123456
Problem: Bayesian optimization stops before budget ends
```

## Submitting Pull Requests

Before submitting a pull request:

1. Make sure the code runs without errors.
2. Check that the README is updated if behavior changes.
3. Confirm that new experiment outputs are not unnecessarily committed.
4. Keep the pull request focused on one clear improvement.
5. Describe what changed and why.

A good pull request description should include:

* Summary of changes
* Experiment folder affected
* How the change was tested
* Any limitations or follow-up work

## Output Files and Logs

Please do not commit large generated outputs unless they are necessary for documentation.

Avoid committing:

* SLURM output folders
* Large model checkpoints
* Temporary cache files
* Joblib temporary files
* Python cache folders
* Local environment files

Examples of files that should usually be ignored:

```text
__experi_*_output/
*.out
*.err
__pycache__/
.ipynb_checkpoints/
.joblib/
*.pkl
*.pt
*.pth
.env
```

## Documentation Updates

Documentation changes are welcome.

When updating documentation, please keep the language clear and direct. Include commands that can be copied and run, especially for SLURM execution.

## Support

For questions, open an issue or contact the maintainer listed in the repository README.

## License

By contributing to this repository, you agree that your contributions will be licensed under the same license as the project.

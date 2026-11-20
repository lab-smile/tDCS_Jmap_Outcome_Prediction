# tDCS J-map Outcome Prediction (ACT trial)

![Python](https://img.shields.io/badge/Python-3.13-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)
[![DOI](https://img.shields.io/badge/DOI-10.3389%2Ffnhum.2026.1847131-blue)](https://doi.org/10.3389/fnhum.2026.1847131)

Official code for the paper:

> **Machine learning and individual variability in electrical field characteristics predict tDCS treatment response for anxiety in older adults in the ACT trial**
> Junfu Cheng, Skylar E. Stolte, Zeyun Zhao, Andrew O'Shea, Aprinda Indahlastari, Adam J. Woods, Ruogu Fang
> *Frontiers in Human Neuroscience*, 20:1847131 (2026). <https://doi.org/10.3389/fnhum.2026.1847131>

---

## Where to find what

This repository bundles code for **several related tasks** (MATLAB feature extraction, the main classification pipeline, feature-type comparisons, regression, and figure scripts). It therefore does **not** follow the lab's recommended `src/ scripts/ configs/` layout; each top-level folder is self-contained (see [Repository map](#repository-map)).

| Looking for… | Go to |
| --- | --- |
| Paper title, authors, venue, DOI | [Publication](#publication) |
| How to cite (text + BibTeX) | [Citation](#citation) |
| **Environment Documentation** (Python/package versions, CUDA, hardware, external software) | [**Environment Documentation**](#environment-documentation) |
| **Reproducibility Verification** (training/inference scripts, hyperparameters, random seeds, figure → script map, verification status) | [**Reproducibility Verification**](#reproducibility-verification) |
| Installation | [Installation](#installation) |
| Dataset description and data availability | [Dataset description](#dataset-description) |
| Preprocessing (MATLAB/SPM12 → MNI-space J-maps) | [Data preprocessing instructions](#data-preprocessing-instructions) |
| Training / inference / reproducing paper numbers | [Training](#training-instructions) · [Inference](#inference-instructions) · [Reproduction](#reproduction-instructions) |
| Expected outputs | [Expected outputs](#expected-outputs) |
| Paths you must set for your system | [Site-specific paths](#site-specific-paths-you-must-set) |
| Contact, license, maintenance, support | [Contact](#contact) · [License](#license) · [Maintenance status](#maintenance-status) · [Support policy](#support-policy) |

---

## Publication

| Field | Value |
| --- | --- |
| Title | Machine learning and individual variability in electrical field characteristics predict tDCS treatment response for anxiety in older adults in the ACT trial |
| Authors | Junfu Cheng, Skylar E. Stolte, Zeyun Zhao, Andrew O'Shea, Aprinda Indahlastari, Adam J. Woods, Ruogu Fang (corresponding author) |
| Journal | *Frontiers in Human Neuroscience*, section Brain Imaging and Stimulation |
| Volume / article | Volume 20, article 1847131 |
| Published | 31 July 2026 |
| Article type | Original Research |
| DOI | [10.3389/fnhum.2026.1847131](https://doi.org/10.3389/fnhum.2026.1847131) |
| Full text | <https://www.frontiersin.org/journals/human-neuroscience/articles/10.3389/fnhum.2026.1847131/full> |
| Article license | CC BY 4.0 |
| Parent trial | Augmenting Cognitive Training in Older Adults (ACT), NCT02851511 |
| Preprint (arXiv) | Not posted |
| Hugging Face model | Not released (models are retrained for every seed; no checkpoint is distributed) |

**Summary.** Machine-learning models were trained on MRI-derived, finite-element-modelled current-density maps (J-maps) to predict state-anxiety improvement after active tDCS paired with cognitive training in older adults with moderate/severe baseline anxiety. The paper reports a mean balanced accuracy of **82%** with repeated mixed-site nested cross-validation and **72%** with repeated cross-site held-out validation (train UF, test UA). SHAP analysis identified current density in the posterior temporal fusiform cortex and the inferior temporal gyrus (posterior and temporooccipital divisions) as the most influential predictors.

## Citation

If you use this code, please cite:

> Cheng J, Stolte SE, Zhao Z, O'Shea A, Indahlastari A, Woods AJ and Fang R (2026) Machine learning and individual variability in electrical field characteristics predict tDCS treatment response for anxiety in older adults in the ACT trial. *Front. Hum. Neurosci.* 20:1847131. doi: 10.3389/fnhum.2026.1847131

```bibtex
@article{cheng2026machine,
  title     = {Machine learning and individual variability in electrical field characteristics predict {tDCS} treatment response for anxiety in older adults in the {ACT} trial},
  author    = {Cheng, Junfu and Stolte, Skylar E. and Zhao, Zeyun and O'Shea, Andrew and Indahlastari, Aprinda and Woods, Adam J. and Fang, Ruogu},
  journal   = {Frontiers in Human Neuroscience},
  volume    = {20},
  pages     = {1847131},
  year      = {2026},
  publisher = {Frontiers Media SA},
  doi       = {10.3389/fnhum.2026.1847131},
  url       = {https://www.frontiersin.org/journals/human-neuroscience/articles/10.3389/fnhum.2026.1847131/full},
  issn      = {1662-5161}
}
```

---

## Project overview

```
ROAST 11-tissue FEM outputs (subject space: T1.nii, *_e.nii, *_Jbrain.nii)
        │  read_matlab_from_skylar/   (MATLAB + SPM12)
        ▼
θ / φ direction maps, Jx/Jy/Jz components, |J| magnitude  →  normalized to MNI152
        │
        ├── ml_clinical_trial/                    main classification pipeline (paper results, SHAP)
        ├── <repo root> run_experiment.py         feature-type comparisons (θ/φ, Jxyz, |J|, …)
        └── delos_fusion_j_map_regression_plus_v1/ unified classification + regression runner
        ▼
j_map_range/   ROI current-density box plots and current-density slice figures
```

Classification pipeline (paper): `J-map flatten + StandardScaler → SMOTE → PCA → Welch t-test → mRMR → Random Fourier Features (RBFSampler) → logistic regression trained with SGD`.

## Repository map

| Folder / file | Task | Entry point(s) | Own docs |
| --- | --- | --- | --- |
| [`read_matlab_from_skylar/`](read_matlab_from_skylar/) | MATLAB: validate `_Jbrain.nii`, compute θ/φ and Jx/Jy/Jz from ROAST outputs, normalize to MNI152 with SPM12 | `batch_make_*.m`, `registration_all_auto_part_*.m` | [`readme.md`](read_matlab_from_skylar/readme.md) |
| [`ml_clinical_trial/`](ml_clinical_trial/) | **Main classification pipeline used for the paper**: cross-site and mixed-site validation, SHAP, SHAP brain figures, ROC/PR curves | `main.py` (cross-site), `main_internal.py` (mixed-site), `main_brain_shap_plotter_cortical*.py`, `main_roc_pr_curve_plot.py` | [`README.md`](ml_clinical_trial/README.md) |
| Repository root (`config.py`, `data_io.py`, `features.py`, `train_eval.py`, `metrics.py`, `plots.py`, `report.py`, `run_experiment.py`, `run_main.bash`) + [`ml_clinical_act_jmap/`](ml_clinical_act_jmap/) | Feature-type comparison (θ, φ, θ+φ, Jx/Jy/Jz, \|J\|, combinations) with cross-site / mixed-site splits and bootstrap CIs | `python -m j_map_tdcs_brain.run_experiment`, `run_main.bash` | [Module details](#module-details-root-level-feature-type-pipeline) |
| [`delos_fusion_j_map_regression_plus_v1/`](delos_fusion_j_map_regression_plus_v1/) | Unified runner: input configs × {classification, regression} × {cross_site, mix_site} × {default, Bayesian-optimized}, 20 seeds | `main.py` | docstring in `main.py` |
| [`j_map_range/`](j_map_range/) | Figure 6 box plots (current density in the top-7 SHAP ROIs); current-density slice figures | `roi_current_density_boxplot.py`, `plot_j_map_main.py` | docstrings |
| `LICENSE`, `CONTRIBUTING.md` | License and contribution guide | | |

---

## Installation

```bash
# 1) Clone into a folder named j_map_tdcs_brain
#    (the root-level pipeline is run as the Python package `j_map_tdcs_brain`)
git clone https://github.com/lab-smile/tDCS_Jmap_Outcome_Prediction.git j_map_tdcs_brain
cd j_map_tdcs_brain

# 2) Create the Python environment (versions pinned to the environment that
#    produced the published results; see "Environment Documentation")
conda create -n jmap_tdcs python=3.13 -y
conda activate jmap_tdcs
pip install numpy==2.2.0 pandas==2.2.3 scipy==1.15.3 scikit-learn==1.6.1 \
    imbalanced-learn==0.13.0 mrmr-selection==0.2.8 shap==0.47.2 \
    nilearn==0.11.1 nibabel==5.3.2 matplotlib==3.10.0 seaborn==0.13.2 \
    statsmodels==0.14.4 xgboost==3.0.2 scikit-optimize==0.10.2 \
    sklearn-genetic-opt==0.12.0 optuna==4.3.0 joblib==1.4.2 \
    openpyxl==3.1.5 msoffcrypto-tool
```

Alternatively, `ml_clinical_trial/environment.yml` can be used with conda (`conda env create -f ml_clinical_trial/environment.yml -n <env_name>`); see the note in [Environment Documentation](#environment-documentation) on how it differs from the paper environment. HiPerGator users can build it with `sbatch ml_clinical_trial/bash_scripts/install_env.sbatch <path/to/environment.yml> <env_name>`.

For preprocessing you additionally need **MATLAB** with **SPM12** on the MATLAB path (see [Data preprocessing instructions](#data-preprocessing-instructions)).

---

## Environment Documentation

> This section covers the post-publication checklist items *Conda environment exported*, *environment.yml uploaded*, *Package versions documented*, *CUDA version documented*, *Hardware requirements documented*, and *Installation tested on a clean machine*.

### Python environment used for the published results

The published results were produced on UF HiPerGator with the conda environment `my_env_2025_4_25` (activated by `ml_clinical_trial/bash_scripts/run_ml_clinical_mini.sbatch`). Versions below are the versions importable in that environment:

| Package | Version | Used for |
| --- | --- | --- |
| Python | 3.13.2 | |
| numpy | 2.2.0 | |
| pandas | 2.2.3 | |
| scipy | 1.15.3 | Welch's t-test |
| scikit-learn | 1.6.1 | PCA, RBFSampler (Random Fourier Features), SGDClassifier, CV |
| imbalanced-learn | 0.13.0 | SMOTE |
| mrmr-selection | 0.2.8 | mRMR feature selection |
| shap | 0.47.3.dev11 (nearest release: 0.47.2) | SHAP attributions |
| nilearn | 0.11.1 | Harvard–Oxford atlas, MNI152 template, brain plots |
| nibabel | 5.3.2 | NIfTI I/O |
| matplotlib / seaborn | 3.10.0 / 0.13.2 | figures |
| statsmodels | 0.14.4 | statistics |
| xgboost | 3.0.2 (CPU build) | optional classifiers |
| scikit-optimize | 0.10.2 | `BayesSearchCV` (root-level pipeline) |
| sklearn-genetic-opt | 0.12.0 | optional genetic search |
| optuna | 4.3.0 | Bayesian optimization (TPE) in `delos_fusion_j_map_regression_plus_v1/` |
| joblib | 1.4.2 | parallelism |
| openpyxl, msoffcrypto-tool | 3.1.5, – | reading the (encrypted) ACT Excel export |

Note: the manuscript lists SciPy v1.16.0; the archived environment contains SciPy 1.15.3.

### `ml_clinical_trial/environment.yml`

`ml_clinical_trial/environment.yml` is a `conda env export` of a separate environment, `clinical_ml` (Python 3.11.13, scikit-learn 1.7.1, imbalanced-learn 0.14.0, scipy 1.16.1). That environment is the one activated by `run_main.bash`. It does **not** list `shap`, `nilearn`, `nibabel` or `mrmr-selection`; install those with pip if you use it. To match the paper exactly, use the pinned versions above.

### CUDA / GPU

Not applicable. All code is CPU-only (no PyTorch/TensorFlow; XGBoost is the CPU build).

### Hardware

| Item | Value |
| --- | --- |
| System used | UF HiPerGator (Linux, SLURM) |
| Reported in the paper | 50 CPU cores, 175 GB RAM |
| SLURM request, `ml_clinical_trial/bash_scripts/run_ml_clinical_mini.sbatch` | 64 CPUs, 200 GB RAM, 8 h |
| SLURM request, `run_main.bash` (root-level pipeline) | 32 CPUs, 400 GB RAM, 6 h |
| GPU | not required |

Voxel-wise features are flattened MNI152 volumes, so memory use grows with the number of channels (e.g. `FEATURE_MODE="all"` uses six channels per voxel).

### External (non-Python) software

| Software | Version | Used for |
| --- | --- | --- |
| ROAST (11 tissue classes) | not recorded | Upstream FEM simulation producing `T1_tDCSLAB_e.nii` and `T1_tDCSLAB_Jbrain.nii` (not part of this repo) |
| MATLAB | not recorded | `read_matlab_from_skylar/` scripts |
| SPM12 | SPM12 | Normalization to MNI152 (`registration_all_auto_part_*.m`) |
| MRIcroGL | v26.0.1 | Rendering of the SHAP brain map (paper Figure 5) |

### Installation test status

| Check | Status |
| --- | --- |
| Pinned pip recipe (above) installs and all packages import in a fresh virtual environment | ✅ Verified 2026-11-20 on HiPerGator (Python 3.13.2) |
| Full installation + run on a clean machine by an independent user | ⬜ Pending |

---

## Dataset description

* **Source:** ACT trial (NCT02851511), older adults (65–89 y) recruited at the University of Florida (UF) and the University of Arizona (UA).
* **Analytic sample:** participants with moderate/severe baseline state anxiety (STAI-state) who received active tDCS + cognitive training and had a valid T1 MRI: **20 participants (11 UF, 9 UA)**.
* **Label:** greater vs. lesser improvement, split at the sample median decrease in STAI-state score (`responder_criteria='above_median_decrease_in_severe'`).
* **Inputs:** subject-specific current-density maps from ROAST finite-element modelling, normalized to MNI152:
  * `wT1_tDCSLAB_Jbrain.nii`: current-density magnitude \|J\| (A/m²)
  * `wT1_tDCSLAB_theta_fromE_maskJbrain.nii`, `wT1_tDCSLAB_phi_fromE_maskJbrain.nii`: direction angles (radians)
  * `wT1_tDCSLAB_Jbrain_{x,y,z}_fromEmag.nii`: Cartesian current-density components
* **Cohort split:** UF participant IDs start with `1`/`2`, UA IDs with `3`. The fixed subject lists and labels are in `config.py` and `delos_fusion_j_map_regression_plus_v1/constants.py`.
* **Atlases:** Harvard–Oxford (via nilearn; SHAP ROI aggregation and Figure 6); Hammers n30r83 (`ml_clinical_trial/hammers_atlas/`; region mapping inside `JmapACTPreprocessor`).

**Data availability.** Imaging data, simulations and clinical variables are **not included** in this repository. ACT trial data are managed under the data-sharing agreement with the NIA. Data requests should go to the parent-trial investigators, as described in the paper's Data Availability Statement. Do not commit participant data (NIfTI files, `data_generation_log/`, `results_*`) to this repository.

---

## Data preprocessing instructions

### Step 1 — Feature maps (MATLAB + SPM12), `read_matlab_from_skylar/`

The registration scripts treat the folder that contains them (`read_matlab_from_skylar/`) as the working directory. Subject folders `read_matlab_from_skylar/<subject_id>/` are created there. In MATLAB:

```matlab
cd <repo>/read_matlab_from_skylar
% 0) set dataBaseDir in the batch scripts to your ROAST output root (see "Site-specific paths")
count_jbrain_dim                              % Step 1: check _Jbrain.nii holds brain-only voxels
batch_make_theta_phi_to_workspace             % Step 2: θ/φ from _e.nii masked by _Jbrain.nii (subject space)
supplement_of_batch_make_theta_phi_to_workspace   % only for re-simulated subject 301428
batch_make_Jxyz_from_emag_and_Jbrain          % Step 2b: Jx/Jy/Jz (subject space)
batch_make_Jxyz_from_emag_and_Jbrain_SPECIAL_301428  % only for subject 301428
registration_all_auto_part_theta              % Step 3: θ → MNI152 (SPM12)
registration_all_auto_part_phi                % Step 3: φ → MNI152
registration_all_auto_part_Jbrain_components  % Step 3b: |J|, Jx, Jy, Jz → MNI152
```

Outputs are the `w*`-prefixed MNI152 files in each `<subject_id>/` folder. Definitions, voxel masks and coordinate conventions are documented in [`read_matlab_from_skylar/readme.md`](read_matlab_from_skylar/readme.md).

### Step 2 — Tabular dataset for `ml_clinical_trial/`

From the repository root, after setting the ACT Excel path and the MNI J-map folder in `ml_clinical_trial/ml_clinical_act_jmap/jmap_act_data_processor.py`:

```bash
mkdir -p data_generation_log/act_data data_generation_log/generated_table
python -m ml_clinical_trial.ml_clinical_act_jmap.jmap_act_data_processor
```

This writes `data_generation_log/act_data/act_data_generated.csv` and `act_data_dict_generated.csv`. The training scripts read them from `<repo>/data_generation_log/act_data/`. `data_generation_log/generated_table/` must exist under the directory you run from, because importing `ml_clinical_trial` writes `metadata_stai_bdi_ii.csv` there (`ml_clinical_trial/meta_df.py`).

---

## Training instructions

All commands are run from the repository root unless stated otherwise.

### A. Main classification pipeline (paper results), `ml_clinical_trial/`

| Validation | Command | Settings used for the paper |
| --- | --- | --- |
| Cross-site held-out (train UF → test UA) | `python -m ml_clinical_trial.main` | 27 seeds (list in [Random seeds](#random-seeds)); SHAP tables saved per seed |
| Mixed-site nested CV (UF+UA pooled, 5-fold) | `python -m ml_clinical_trial.main_internal` | set `random_state = 82` and `num_training_repetition = 20` in `main_internal.py` (seeds 82–101 × 5 folds = 100 evaluations) |

On HiPerGator: `sbatch ml_clinical_trial/bash_scripts/run_ml_clinical_mini.sbatch <repo_root>` (runs `main_internal`).

### B. Feature-type comparison, repository root package

Edit `FEATURE_MODE` (`theta`, `phi`, `concat`, `tp_jmag`, `jxyz`, `jmag`, `all`) and `SPLIT_MODE` (`manual`, `cross_site`, `mixed_site`) in `config.py`, then run from the **parent** directory of the repository:

```bash
cd ..                                   # folder that contains j_map_tdcs_brain/
python -m j_map_tdcs_brain.run_experiment
# or on SLURM:
sbatch j_map_tdcs_brain/run_main.bash "$(pwd)"
```

### C. Unified classification + regression runner, `delos_fusion_j_map_regression_plus_v1/`

```bash
cd delos_fusion_j_map_regression_plus_v1
python main.py --tasks classification regression \
               --split_modes cross_site mix_site \
               --input_configs magnitude theta_phi magnitude_theta_phi jxyz magnitude_demo \
               [--no_bayes | --bayes_only] [--parallel_cv] [--parallel_reps]
```

Paths, seeds and Bayesian-optimization settings are in `constants.py`.

## Inference instructions

No pretrained checkpoint is distributed (and no Hugging Face model). Models are retrained for every seed, and inference on held-out participants runs inside each training run:

* `ml_clinical_trial/`: `ModelEvaluator.evaluate()` and `collect_performance_metrics()` in `model_evaluator.py` score the fitted pipeline on the held-out set (probability threshold 0.5). `get_coordinate_table_mni_shap()` computes SHAP on the same held-out set.
* Root-level pipeline: `fit_best_and_predict()` / `predict_proba_positive()` in `train_eval.py`.

To score new participants, preprocess their J-maps exactly as in [Data preprocessing](#data-preprocessing-instructions) and pass them as the test split.

## Reproduction instructions

| Paper result | How to reproduce | Expected value (from the archived run) |
| --- | --- | --- |
| Mixed-site nested CV, balanced accuracy 82% | `python -m ml_clinical_trial.main_internal` with `random_state = 82`, `num_training_repetition = 20` | `table/performance/performance_summary_stats.csv`: Balanced_Accuracy mean 0.823 (n = 100), ROC_AUC mean 0.887 |
| Cross-site held-out, balanced accuracy 72% | `python -m ml_clinical_trial.main` (first 27 seeds of `merged_list`) | `table/performance/performance_summary_stats.csv`: Balanced_Accuracy 0.722 ± 0.060 (n = 27), ROC_AUC mean 0.735 |
| Figure 5: top-10 ROI SHAP map | Cross-site run → `seed_154/shap_table.csv` → set `shap_csv_path` and run `python -m ml_clinical_trial.main_brain_shap_plotter_cortical` (top-10 cortical ROIs). `python -m ml_clinical_trial.main_brain_shap_plotter_cortical_subcortical` writes the ROI table and the \|SHAP\| volume `figure_brain/abs_arr_visualize.nii.gz` for MRIcroGL rendering | ranking committed in `ml_clinical_trial/figure_brain/roi_contrib_shap_mean.csv` |
| Figure 6: current density in the top-7 SHAP ROIs by group | `cd j_map_range && python roi_current_density_boxplot.py` | `j_map_range/figures/roi_boxplots/top7_roi_current_density.{png,pdf,svg}` and `roi_range/roi_value_range.{csv,txt}` (committed) |
| ROC / PR curves aggregated over seeds | set `experiment_dir` in `ml_clinical_trial/main_roc_pr_curve_plot.py` and run it as a module | curve figures under `<experiment_dir>/figure/curves/` |

## Expected outputs

| Pipeline | Location | Contents |
| --- | --- | --- |
| `ml_clinical_trial` | `./results_<YYYY_MM_DD_HH_MM_SS>/<random_state>/` | `seed_<s>/` (cross-site) or `seed_<s>_fold_<k>/` (mixed-site) with `performance_report.csv` and `shap_table.csv` (cross-site); `figure/curves/` (ROC/PR, per-run metrics); `figure/balanced_acc/` (histogram); `table/performance/per_run_performance.csv`, `performance_summary_stats.csv` (mean, SD, SE, 95% CI, median, IQR); `output_<timestamp>.log` |
| SHAP plotters | `figure_brain/` | `roi_contrib_shap_bar.png`, `roi_shap_heatmap.png`, `roi_contrib_shap_mean.csv`, `abs_arr_visualize.nii.gz`, `abs_arr_hist.png` |
| Root-level pipeline | `<repo>/results<YYYY-MM-DD-HH-MM-SS>/` | `test_roc.{png,svg,pdf}`, `test_pr.{png,svg,pdf}`, `test_metrics.csv` (mean ± SD, 95% CI), `summary_report.pdf` |
| `delos_fusion_j_map_regression_plus_v1` | `output/{task}/{split_mode}/{input_config}/{bo_mode}/` | `*_per_repetition.csv`, `*_summary.csv`; `output/all_summaries.csv` |
| `j_map_range` | `j_map_range/figures/roi_boxplots/` | `top7_roi_current_density.{png,pdf,svg}`, `roi_range/` |
| MATLAB | `read_matlab_from_skylar/<subject_id>/` | subject-space θ/φ/Jxyz and `w*` MNI152 NIfTI files |

---

## Reproducibility Verification

> This section covers the post-publication checklist items *Training scripts provided*, *Inference scripts provided*, *Example notebook provided*, *Hyperparameters documented*, *Random seeds documented*, *All major figures reproducible from released code*, and *Independent lab member successfully reproduces key results*.

| Checklist item | Status | Where |
| --- | --- | --- |
| Training scripts provided | ✅ | `ml_clinical_trial/main.py`, `ml_clinical_trial/main_internal.py`, `run_experiment.py`, `delos_fusion_j_map_regression_plus_v1/main.py`; SLURM: `ml_clinical_trial/bash_scripts/run_ml_clinical_mini.sbatch`, `run_main.bash` ([Training](#training-instructions)) |
| Inference scripts provided | ✅ (in-run evaluation; no standalone checkpoint) | `ml_clinical_trial/model_evaluator.py`, `train_eval.py` ([Inference](#inference-instructions)) |
| Example notebook provided | N/A | no notebooks are part of the workflow |
| Hyperparameters documented | ✅ | [Hyperparameters](#hyperparameters) |
| Random seeds documented | ✅ | [Random seeds](#random-seeds) |
| Major figures reproducible from released code | ✅ Figures 5–6, ROC/PR curves | [Reproduction instructions](#reproduction-instructions) |
| Independent lab member reproduces key results | ⬜ Pending | |

### Hyperparameters

**`ml_clinical_trial/` (paper pipeline; `pipeline_generator` in `main.py` and `main_internal.py`):**

| Step | Class | Setting |
| --- | --- | --- |
| Volume features | `JmapACTPreprocessor` | `strategy="flatten"`, `keep_channel_axis=True`, `scale_volume=True` (StandardScaler) |
| Oversampling | `SafeSMOTE` | `sampling_strategy=1.0`, `k_neighbors=2` |
| Dimensionality reduction | `PCAWithNames` | `n_components=0.95` (95% explained variance) |
| Univariate filter | `WelchTTestSelector` | `p_thresh=1e-4`, `min_k_if_empty=2000`, `cap_after_t=15000` |
| mRMR | `MRMRSelector` | `frac_for_topk=0.01`, `min_topk=10`, `max_topk=20` |
| Random Fourier Features | `RBFSampler` | `gamma=1.0`, `n_components=10` |
| Classifier | `SGDClassifier` | `loss="log_loss"`, `max_iter=1000` |
| Decision threshold | — | 0.5 |

These values were selected by **Bayesian optimization** (Gaussian-process surrogate with a Matérn kernel, as described in the paper) and are fixed in `pipeline_generator`. The final cross-site and mixed-site runs fit the pipeline with these optimized values: `ModelTrainer.train_by_grid_search_cv()` (`ml_clinical_trial/model_trainer.py`) runs no further search, and the `param_grid` passed to it is only logged.

**Root-level pipeline (`train_eval.py`, `config.py`):** same pipeline steps (`RBFSampler n_components=300` by default). Search space: `mrmr__max_topk` 10–500, `rbf__gamma` 1e-3–1e1 (log), `rbf__n_components` 100–1000. Strategy `HYPEROPT_STRATEGY` ∈ {`grid`, `random`, `bayes`, `genetic`}. `bayes` uses scikit-optimize `BayesSearchCV`, whose default surrogate is a Gaussian process with a Matérn kernel. Inner CV: `RepeatedStratifiedKFold(n_splits=N_SPLITS=3, n_repeats=N_REPEATS=1)`, scored by ROC AUC. Outer CV: `OUTER_N_SPLITS=5`.

**`delos_fusion_j_map_regression_plus_v1/`:** Optuna TPE sampler, `BAYES_OPT_N_ITER=10`, `BAYES_OPT_CV_FOLDS=3` (`constants.py`, `bayes_opt_helper.py`).

### Random seeds

| Experiment | Seeds | Where set |
| --- | --- | --- |
| Cross-site held-out (paper, archived run `results_2025_09_11_12_25_06`) | 167, 177, 180, 143, 152, 154, 155, 156, 124, 126, 127, 128, 134, 102, 107, 109, 112, 87, 90, 67, 71, 72, 75, 46, 49, 53, 61 (27 seeds) | `merged_list` in `ExperimentManager.run_grid_search_opt_cv_experiment_in_severe_state_anxiety_in_jmap` (`ml_clinical_trial/experiment_manager.py`). The list in the code also contains 16 later seeds (484–579); the paper run used the first 27 |
| **SHAP (paper Figure 5, Figure 6 ROI selection)** | **seed 154** of the cross-site run (balanced accuracy 0.90 on the UA test set) | `shap_csv_path=.../seed_154/shap_table.csv` in `main_brain_shap_plotter_cortical.py`, `main_brain_shap_plotter_cortical_subcortical.py`, `brain_figure_plotter.py`; the top-7 ROIs of this table are `TOP7_ROIS` in `j_map_range/roi_current_density_boxplot.py` |
| Mixed-site nested CV (paper, archived run `results_2025_10_11_04_19_01`) | 82–101 (20 seeds) × 5 outer folds | `random_state`, `num_training_repetition` in `ml_clinical_trial/main_internal.py` |
| Root-level pipeline | `RANDOM_STATE=42`, `BOOTSTRAP_SEED=40`, `BOOTSTRAP_N=20` | `config.py` |
| `delos_fusion_j_map_regression_plus_v1` | 42–61 (`BASE_SEED=42`, `NUM_REPETITIONS=20`) | `constants.py` |

Every stochastic step (SMOTE, PCA, RBFSampler, SGD, CV splitters) receives the per-run seed through `pipeline_generator(..., random_state)`.

---

## Site-specific paths you must set

Paths inside this repository are resolved relative to the repository. The following absolute paths point to participant data or archived results on HiPerGator that cannot be distributed. Edit them for your system:

| File | Variable | Points to |
| --- | --- | --- |
| `read_matlab_from_skylar/batch_make_theta_phi_to_workspace.m`, `batch_make_Jxyz_from_emag_and_Jbrain.m`, `count_jbrain_dim.m` | `dataBaseDir`, `niiFile` | ROAST outputs |
| `read_matlab_from_skylar/supplement_of_batch_make_theta_phi_to_workspace.m`, `batch_make_Jxyz_from_emag_and_Jbrain_SPECIAL_301428.m` | `dataBaseDir` | re-simulated ROAST output of subject 301428 |
| `ml_clinical_trial/ml_clinical_act_jmap/jmap_act_data_processor.py` | `root_dir`, `input_dataset_file_path` | MNI J-maps, ACT Excel export |
| `ml_clinical_trial/main_brain_shap_plotter_cortical.py`, `main_brain_shap_plotter_cortical_subcortical.py`, `brain_figure_plotter.py` | `shap_csv_path` | `<results>/seed_154/shap_table.csv` |
| `ml_clinical_trial/main_roc_pr_curve_plot.py`, `model_evaluator.py` (`__main__`) | `experiment_dir` | a results folder |
| `ml_clinical_trial/brainviz.py`, `brainviz_pro.py` | `output_dir` / `out_dir` | figure output folder |
| `j_map_range/plot_j_map_main.py` | `base_dir`, `output_dir` | MNI J-maps, figure output |
| `j_map_range/roi_current_density_boxplot.py` | `JMAP_BASE_DIR` | MNI J-maps |
| `delos_fusion_j_map_regression_plus_v1/constants.py` | `EHR_FILE_PATH`, `JMAP_BASE_DIR` | ACT Excel export, MNI J-maps |

---

## Module details: root-level feature-type pipeline

<details>
<summary>θ/φ, Jxyz and |J| feature modes, data layout and split-specific uncertainty estimation (click to expand)</summary>

### Input features

The pipeline reads per-subject MNI152 maps from `<repo>/read_matlab_from_skylar/<subject_id>/` (`OLD_WORKING_DIR` in `config.py`), produced by the MATLAB step:

```
<subject_id>/
├── wT1_tDCSLAB_theta_fromE_maskJbrain.nii
├── wT1_tDCSLAB_phi_fromE_maskJbrain.nii
├── wT1_tDCSLAB_Jbrain.nii
├── wT1_tDCSLAB_Jbrain_x_fromEmag.nii
├── wT1_tDCSLAB_Jbrain_y_fromEmag.nii
└── wT1_tDCSLAB_Jbrain_z_fromEmag.nii
```

These maps are derived from ROAST 11 simulations, masked by `_Jbrain.nii`, and normalized to MNI152 with SPM12. Angles are stored in radians and treated as scalar fields (no vector reorientation).

### Feature modes (`FEATURE_MODE` in `config.py`)

Channel order in the loaded 5-D array: `0 θ, 1 φ, 2 Jx, 3 Jy, 4 Jz, 5 |J|`.

| Mode | Channels | Volume shape per subject |
| --- | --- | --- |
| `theta` | θ | `(X, Y, Z, 1)` |
| `phi` | φ | `(X, Y, Z, 1)` |
| `concat` | θ + φ | `(X, Y, Z, 2)` |
| `tp_jmag` | θ + φ + \|J\| | `(X, Y, Z, 3)` |
| `jxyz` | Jx + Jy + Jz | `(X, Y, Z, 3)` |
| `jmag` | \|J\| | `(X, Y, Z, 1)` |
| `all` | θ + φ + Jx + Jy + Jz + \|J\| | `(X, Y, Z, 6)` |

Volumes are wrapped into a DataFrame with one row per subject and the volume in column `jmap_tp1`, which is the input format of `JmapACTPreprocessor` (`keep_channel_axis=True`):

```python
X_train_df = as_jmap_dataframe(train_ds.X, train_ds.ids, mode="concat")
```

The ML stage does not recompute θ or φ. Volumes are flattened (optionally PCA-reduced or Hammers-region-aggregated), and `StandardScaler` is always applied after voxel extraction.

### Split modes (`SPLIT_MODE` in `config.py`)

* `manual`: fixed `TRAIN_IDS_*` / `TEST_IDS_*`.
* `cross_site`: one site (`CROSS_SITE_TEST_SITE`, default UA) is the independent test set; hyperparameters are tuned with nested CV on the training site only.
* `mixed_site`: UF + UA pooled; `BOOTSTRAP_N` repetitions of `OUTER_N_SPLITS`-fold nested stratified CV; all predictions are out-of-fold (OOF).

Metrics for every mode: AUROC, AUPRC, balanced accuracy, weighted F1 and MCC, with threshold-based metrics at probability 0.5.

### Cross-site uncertainty (non-parametric bootstrap of the held-out test set)

Draw $B = \text{BOOTSTRAP}_{N}$ bootstrap samples with replacement, compute each metric $M^{(b)}$, and report

```math
\mu = \frac{1}{B} \sum_{b=1}^{B} M^{(b)}, \qquad
\sigma = \sqrt{\frac{1}{B-1} \sum_{b=1}^{B} \left(M^{(b)} - \mu\right)^2}, \qquad
\text{CI}_{95\%} = \left[Q_{0.025}\left(\{M^{(b)}\}\right),\; Q_{0.975}\left(\{M^{(b)}\}\right)\right]
```

The CI reflects the sampling variability of the fixed, independent test cohort.

### Mixed-site uncertainty (two aggregation levels, both reported)

1. **Repetition level (OOF pooled):** each subject has one OOF prediction per repetition $r$; metric $M^{(r)}$ is computed per repetition.

```math
\mu_{\text{rep}} = \frac{1}{R} \sum_{r=1}^{R} M^{(r)}, \qquad
\sigma_{\text{rep}} = \sqrt{\frac{1}{R-1} \sum_{r=1}^{R} \left(M^{(r)} - \mu_{\text{rep}}\right)^2}, \qquad
\text{CI}_{95\%,\text{rep}} = \left[Q_{0.025}\left(\{M^{(r)}\}\right),\; Q_{0.975}\left(\{M^{(r)}\}\right)\right]
```

2. **Outer-fold level (all repetitions × folds):** metric $M^{(r,k)}$ on outer test fold $k$ of repetition $r$.

```math
\mu_{\text{fold}} = \frac{1}{RK} \sum_{r=1}^{R} \sum_{k=1}^{K} M^{(r,k)}, \qquad
\sigma_{\text{fold}} = \sqrt{\frac{1}{RK-1} \sum_{r=1}^{R} \sum_{k=1}^{K} \left(M^{(r,k)} - \mu_{\text{fold}}\right)^2}, \qquad
\text{CI}_{95\%,\text{fold}} = \left[Q_{0.025}\left(\{M^{(r,k)}\}\right),\; Q_{0.975}\left(\{M^{(r,k)}\}\right)\right]
```

In mixed-site mode the unit of analysis is the repetition or fold, not individual subjects. No subject-level bootstrap is used, and all uncertainty comes from repeated nested CV. `test_metrics.csv` and the PDF report list both levels (mean ± SD and 95% CI per metric).

</details>

---

## Contact

Junfu Cheng, Department of Electrical and Computer Engineering, University of Florida (junfu.cheng@ufl.edu).
Corresponding author of the paper: Ruogu Fang, University of Florida.

## License

Code: [MIT License](LICENSE), © 2026 Junfu Cheng. The article is published under CC BY 4.0. Third-party files in `ml_clinical_trial/hammers_atlas/` (Hammers n30r83 atlas) are subject to their own license terms.

## Funding

See the Funding section of the [paper](https://doi.org/10.3389/fnhum.2026.1847131) (NIH/NIA support of the ACT trial, R01AG054077, among others).

---

# Maintenance Status

**Repository Name:** tDCS J-map Outcome Prediction (`lab-smile/tDCS_Jmap_Outcome_Prediction`)
**Maintainer(s):** Junfu Cheng, Department of Electrical and Computer Engineering, University of Florida, junfu.cheng@ufl.edu
**Status:** Active
**Last Updated:** 2026-11-20

## Status Definitions

- **Active**: Actively developed, bugs fixed, new features added.
- **Maintenance Mode**: Only critical bug fixes and security updates.
- **Deprecated**: No longer recommended; no new features; may be removed in future.
- **Archived**: Read-only repository; no longer maintained.

## Notes

This repository accompanies the Frontiers in Human Neuroscience (2026) paper above and contains the code used to produce its results and figures.

# Support Policy

**Repository Name:** tDCS J-map Outcome Prediction (`lab-smile/tDCS_Jmap_Outcome_Prediction`)
**Maintainer(s):** Junfu Cheng, Department of Electrical and Computer Engineering, University of Florida, junfu.cheng@ufl.edu

## Supported Versions

- Active branches: `main`
- Unsupported branches: none

## Issue Handling

- Bugs or issues in active branches will be addressed within 10 business days.
- Pull requests from the community will be reviewed based on relevance and priority.
- Deprecated or archived repositories may not receive support.

## Contribution Guidelines

- Please refer to `CONTRIBUTING.md` for pull request and issue submission guidelines.
- When reporting a bug, include steps to reproduce, environment details, logs, and error messages.

## Contact

For questions or clarifications, contact Junfu Cheng, Department of Electrical and Computer Engineering, University of Florida, junfu.cheng@ufl.edu.

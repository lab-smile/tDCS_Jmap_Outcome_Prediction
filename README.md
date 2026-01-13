# direction_ml (current-density direction prediction)

This folder contains modular Python code to:

- Load MNI-warped direction maps (`w*theta*.nii`, `w*phi*.nii`) from the *old* workspace
- Train a classifier to predict responder vs non-responder
- Evaluate on the held-out **test set**
- Export:
  - ROC curve: PNG, SVG, PDF
  - PR curve: PNG, SVG, PDF
  - Test metrics table: CSV
  - One PDF summary report (IDs, labels, metrics, curves)

## How to run

1. Make sure your environment can import your custom modules:

- `ml_clinical_act_jmap.jmap_act_preprocessWrapper`
- `ml_clinical_act_jmap.pca_with_names`
- `ml_clinical_act_jmap.safe_smote`
- `ml_clinical_act_jmap.hetero_selector`

2. Install required packages:

```bash
pip install numpy pandas scikit-learn nibabel matplotlib
# optional:
pip install scikit-optimize sklearn-genetic-opt
```

3. Edit paths and settings in `config.py` if needed.

4. Run:

```bash
python -m direction_ml.run_experiment
```

Outputs will appear under:

```
/orange/ruogu.fang/junfu.cheng/SMILE/j_map/j_map_direction/direction_ml/results/
```

---

# new feature-setting option for θ / φ handling

This module performs **machine-learning–based prediction of responder vs non-responder status** using **MNI-registered current density direction maps** derived from ROAST simulations.

The pipeline consumes **θ (zenith)** and **φ (azimuthal)** direction maps in **MNI152 space**, extracted from the previous MATLAB-based preprocessing workflow, and evaluates model performance on a held-out testing set with uncertainty estimation.

---

## Input Features: Directional Maps in MNI Space

### Source of Directional Features

All directional features are read from the **previous working directory**:

```
/orange/ruogu.fang/junfu.cheng/SMILE/j_map/j_map_direction/read_matlab_from_skylar/
```

For each subject, the following files are required:

```
<subject_id>/
├── wT1_tDCSLAB_theta_fromE_maskJbrain.nii
└── wT1_tDCSLAB_phi_fromE_maskJbrain.nii
```

These files are:

* Derived from ROAST 11 electric field simulations
* Masked by `_Jbrain.nii`
* Spatially normalized to **MNI152 space** using SPM12
* Stored in **radians**
* Treated as scalar fields (no vector reorientation applied)

---

## Feature Configuration: θ / φ Handling Options

### Overview

The machine learning pipeline supports **flexible feature configurations** that control how directional information is presented to the model.

This is implemented when constructing the input DataFrame passed to `JmapACTPreprocessor`.

Each subject contributes **one volume per row**, stored as a NumPy array in a DataFrame column (default: `jmap_tp1`).

---

### Feature Modes

The feature mode is controlled by a parameter (`mode`) in the helper function that constructs the DataFrame.

#### Supported Modes

| Mode     | Description                               | Volume Shape per Subject |
| -------- | ----------------------------------------- | ------------------------ |
| `concat` | Concatenate θ and φ as channels (default) | `(X, Y, Z, 2)`           |
| `theta`  | Use θ only                                | `(X, Y, Z, 1)`           |
| `phi`    | Use φ only                                | `(X, Y, Z, 1)`           |

---

### Default (Recommended): Concatenated θ + φ

By default, **θ and φ are concatenated as two channels**, forming a 4D volume per subject:

```
(X, Y, Z, C=2)
```

Channel order is fixed as:

```
channel 0 → θ (zenith angle)
channel 1 → φ (azimuthal angle)
```

This configuration preserves **full directional information** and is recommended for primary analyses.

---

### Alternative: Single-Angle Feature Sets

For ablation studies or interpretability analyses, the pipeline can be run using **θ-only** or **φ-only** features.

In these cases:

* Only one angular component is retained
* The volume still preserves a channel axis (`C=1`)
* All downstream preprocessing (scaling, PCA, feature selection) remains unchanged

---

### Implementation Detail

Directional volumes are wrapped into a pandas DataFrame with the following structure:

| subject_id | jmap_tp1                |
| ---------- | ----------------------- |
| 105256     | NumPy array `(X,Y,Z,C)` |
| 105691     | NumPy array `(X,Y,Z,C)` |
| …          | …                       |

Each cell in `jmap_tp1` contains a **subject-specific 3D or 4D volume**, depending on feature mode.

This design allows:

* Seamless integration with `JmapACTPreprocessor`
* Consistent handling of volumetric features
* Easy switching between feature configurations without altering the ML pipeline

---

### Example Usage

```python
# θ + φ concatenated (default)
X_train_df = as_jmap_dataframe(train_ds.X, train_ds.ids, mode="concat")

# θ only
X_train_df = as_jmap_dataframe(train_ds.X, train_ds.ids, mode="theta")

# φ only
X_train_df = as_jmap_dataframe(train_ds.X, train_ds.ids, mode="phi")
```

---

## Notes on Downstream Processing

* Channel handling is preserved by setting:

  ```python
  keep_channel_axis=True
  ```

  in `JmapACTPreprocessor`.

* Volumes are optionally:

  * Flattened
  * Reduced via PCA
  * Region-aggregated using the Hammers atlas

* Feature scaling is always applied (`StandardScaler`) after voxel extraction.

* No recomputation of θ or φ occurs in the ML stage.

---

## Reproducibility and Traceability

* Original MNI-space direction maps are **read-only**
* Feature mode selection is explicit and logged
* Training, validation, and testing splits are fixed by subject ID
* Outputs include:

  * ROC and PR curves (PNG, SVG, PDF)
  * Bootstrap-based test metrics with 95% CI (CSV)
  * A single PDF summary report aggregating all results

---

## Performance Metrics, Uncertainty Estimation, and Split-Specific Definitions

This pipeline supports multiple data-splitting strategies (`SPLIT_MODE`), each of which implies a **different statistical definition** of performance estimation and uncertainty.
The computation of **mean**, **standard deviation (SD)**, and **95% confidence intervals (CI)** therefore depends on the selected split mode.

The supported split modes are:

* `cross_site`
* `mixed_site`

Below we describe **exactly how metrics are computed and summarized** for each option.

---

### Common Definitions

Across all split modes, the following performance metrics are computed:

* AUROC (area under the ROC curve)
* AUPRC (area under the precision–recall curve)
* Balanced Accuracy
* Weighted F1 score
* Matthews Correlation Coefficient (MCC)

Threshold-based metrics use a **fixed probability threshold of 0.5**.

---

## Cross-Site Split (`SPLIT_MODE = "cross_site"`)

### Design

* One site (e.g., UA) is held out entirely as an **independent test set**
* The remaining site (e.g., UF) is used for training
* Hyperparameters are optimized using **nested cross-validation on the training site only**
* Final evaluation is performed **once** on the held-out test site

### Metric Computation

Let:

* ( y_i ) be the true test labels
* ( \hat{p}_i ) be predicted probabilities on the held-out test set

Metrics are computed on the **entire held-out test set**.

---

### Bootstrap-Based Uncertainty Estimation

To estimate uncertainty, **non-parametric bootstrap resampling** is applied to the held-out test set:

1. Draw ( B = \text{BOOTSTRAP_N} ) bootstrap samples (with replacement)
2. Compute each metric on every bootstrap sample
3. Obtain a bootstrap distribution ( { M^{(b)} }_{b=1}^B )

#### Reported Statistics

* **Mean**
  [
  \mu = \frac{1}{B} \sum_{b=1}^B M^{(b)}
  ]

* **Standard Deviation**
  [
  \sigma = \sqrt{\frac{1}{B-1} \sum_{b=1}^B (M^{(b)} - \mu)^2}
  ]

* **95% Confidence Interval**
  [
  \text{CI}*{95%} =
  \left[
  Q*{0.025}({M^{(b)}}),
  ;
  Q_{0.975}({M^{(b)}})
  \right]
  ]

### Interpretation

* CI reflects **sampling variability of the test cohort**
* This is appropriate because the test set is **independent and fixed**
* This is the classical evaluation setup for generalization to a new site

---

## Mixed-Site Split (`SPLIT_MODE = "mixed_site"`)

### Design

* Participants from all sites (UF + UA) are **pooled**
* No fixed held-out test set is used
* Performance is evaluated using:

  * **BOOTSTRAP_N repetitions**
  * Each repetition uses **OUTER_N_SPLITS-fold nested stratified cross-validation**
* Hyperparameter tuning occurs **inside each outer fold**
* All reported predictions are **out-of-fold (OOF)**

---

### Two Levels of Performance Aggregation

In mixed-site mode, metrics are summarized at **two distinct levels**, both of which are reported.

---

### 1. Repetition-Level Metrics (OOF-Pooled)

For repetition ( r ):

* Each subject receives exactly **one OOF prediction**
* Metrics are computed once using all OOF predictions in that repetition

Let ( M^{(r)} ) denote the metric value for repetition ( r ).

#### Mean Across Repetitions

[
\mu_{\text{rep}} =
\frac{1}{R}
\sum_{r=1}^{R} M^{(r)}
]

#### Standard Deviation Across Repetitions

[
\sigma_{\text{rep}} =
\sqrt{
\frac{1}{R-1}
\sum_{r=1}^{R}
\left(M^{(r)} - \mu_{\text{rep}}\right)^2
}
]

#### 95% Confidence Interval (Percentile-Based)

[
\text{CI}*{95%,\text{rep}} =
\left[
Q*{0.025}({M^{(r)}}),
;
Q_{0.975}({M^{(r)}})
\right]
]

**Interpretation:**
Captures **stability of the full modeling pipeline** across independent CV repetitions.

---

### 2. Outer-Fold Test Metrics (All Repetitions × All Folds)

For each repetition ( r ) and outer fold ( k ):

* Metrics are computed **only on the outer test fold**
* This yields ( R \times K ) metric values

Let ( M^{(r,k)} ) denote the metric from repetition ( r ), fold ( k ).

#### Mean Across All Outer-Fold Evaluations

[
\mu_{\text{fold}} =
\frac{1}{RK}
\sum_{r=1}^{R}
\sum_{k=1}^{K}
M^{(r,k)}
]

#### Standard Deviation Across All Outer-Fold Evaluations

[
\sigma_{\text{fold}} =
\sqrt{
\frac{1}{RK-1}
\sum_{r,k}
\left(M^{(r,k)} - \mu_{\text{fold}}\right)^2
}
]

#### 95% Confidence Interval (Percentile-Based)

[
\text{CI}*{95%,\text{fold}} =
\left[
Q*{0.025}({M^{(r,k)}}),
;
Q_{0.975}({M^{(r,k)}})
\right]
]

**Interpretation:**
Reflects **variability across individual outer test folds**, providing a more granular view of performance dispersion.

---

## Summary Table in Mixed-Site Mode

The final metrics table (`test_metrics.csv`) and PDF report include **both levels**:

| Level                              | Metric | Mean ± SD | 95% CI (Lower – Upper) |
| ---------------------------------- | ------ | --------- | ---------------------- |
| Repetition (OOF pooled)            | AUROC  | …         | …                      |
| Repetition (OOF pooled)            | AUPRC  | …         | …                      |
| …                                  | …      | …         | …                      |
| Outer-fold test (all reps × folds) | AUROC  | …         | …                      |
| Outer-fold test (all reps × folds) | AUPRC  | …         | …                      |
| …                                  | …      | …         | …                      |

---

## Key Methodological Notes

* In **mixed-site**, the unit of analysis is **repetition or fold**, not individual subjects
* Bootstrap resampling of subjects is **not used**
* All uncertainty estimates arise from **repeated nested cross-validation**
* This avoids optimistic bias and preserves strict separation between training and evaluation

---
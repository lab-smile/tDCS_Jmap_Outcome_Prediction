
---

# README

## Project: Validation of `_Jbrain.nii` and Directional Mapping of Electric Field–Derived Current Density

This project validates that `_Jbrain.nii` contains **brain-only voxels**, derives **directional information (θ, φ)** from the electric field (`_e.nii`) constrained to valid brain voxels, and subsequently transforms these direction maps into **MNI152 standard space** for group-level analysis.

In addition, the project derives **Cartesian current density components (Jx, Jy, Jz)** in subject space using electric field direction and current density magnitude.

---

## Step 1: Validate Brain-Only Content of `_Jbrain.nii`

### Purpose

The first step confirms whether the `_Jbrain.nii` files include **only brain voxels**.
This validation ensures that all **non-zero values** in `_Jbrain.nii` correspond exclusively to brain tissue, while non-brain regions (skull, scalp, air) are represented as **zero or NaN**.

---

### Implementation

```

count_jbrain_dim.m

```

---

## Step 2: Generate Direction Maps (θ, φ) from `_e.nii` Masked by `_Jbrain.nii`

### Purpose

After validating `_Jbrain.nii`, directional information is derived from the electric field (`_e.nii`).
Because electric field direction and current density direction are expected to be aligned, the **E-field vector** is used to compute direction, while `_Jbrain.nii` defines valid brain voxels.

Angles are computed **in radians** and stored as voxel-wise volumes.

---

### Direction Definitions

Let the electric field vector be:

\[
\vec{E} = (E_x, E_y, E_z)
\]

* **Zenith angle (θ)**:
  \[
  \theta = \cos^{-1}\left(\frac{E_z}{\sqrt{E_x^2 + E_y^2 + E_z^2}}\right)
  \]

* **Azimuthal angle (φ)**:
  \[
  \varphi = \tan^{-1}\left(\frac{E_y}{E_x}\right)
  \]

---

### Voxel Selection Criteria

Angles are computed **only for voxels that satisfy both conditions**:

* `_Jbrain.nii` voxel value ≠ 0  
* Electric field magnitude (|\vec{E}|) ≠ 0  

All other voxels are set to **NaN**.

---

### Outputs (Subject Space)

```

<working_directory>/<subject_id>/
├── T1.nii
├── T1_tDCSLAB_theta_fromE_maskJbrain.nii
└── T1_tDCSLAB_phi_fromE_maskJbrain.nii

```

---

### MATLAB Implementation

Primary batch processing is implemented in:

```

batch_make_theta_phi_to_workspace.m

```

This script:

* Processes multiple subjects  
* Includes a file-size consistency check to skip corrupted `_e.nii` files  
* Copies `T1.nii` into each subject output folder  
* Ensures consistent geometry using `_Jbrain.nii` as reference  

---

### **Note: Supplementary Reprocessing for Corrected ROAST 11 Outputs**

During batch processing, some subjects (e.g., **subject 301428**) were skipped because their original ROAST 11–generated `_e.nii` files were **truncated or corrupted**, causing MATLAB read errors.

To address this issue:

* A **new ROAST 11 simulation** was generated for the affected subject(s)
* The corrected `_e.nii` and `_Jbrain.nii` files were placed in the data directory
* A supplementary MATLAB script was added:

```

supplement_of_batch_make_theta_phi_to_workspace.m

```

This supplement script:

* Specifically reads the **newly generated ROAST 11 `_e.nii` and `_Jbrain.nii` files**
* Applies the **same direction computation and masking logic** as the main batch script
* Generates θ and φ maps (radians) in subject space
* Saves outputs into the same per-subject folder structure in the working directory

This approach ensures:

* Methodological consistency across subjects  
* Recovery of valid results for subjects affected by input data corruption  
* Clear provenance separating original batch outputs from corrected reprocessing  

The supplement script should be used **only for subjects with regenerated ROAST outputs** and **not** for routine batch processing.

---

## Step 2b: Generate Direction-Resolved Current Density Components (Jx, Jy, Jz)

### Purpose

In addition to angular direction maps (θ, φ), this step derives **vector-resolved current density components** within the brain by combining:

* **Current density magnitude** from `_Jbrain.nii`
* **Directional information** from the electric field vector (`_e.nii`)

This produces voxel-wise **Cartesian components of current density** aligned with the local electric field direction:

* `Jx`, `Jy`, `Jz`

All computations are restricted to **valid brain voxels** defined by `_Jbrain.nii`.

---

### Conceptual Model

For each voxel, let:

```

E = (Ex, Ey, Ez)
|E| = sqrt(Ex² + Ey² + Ez²)
û = E / |E|
J = Jbrain

```

Then:

```

Jx = J · (Ex / |E|)
Jy = J · (Ey / |E|)
Jz = J · (Ez / |E|)

```

---

### Coordinate System and Anatomical Interpretation

All NIfTI volumes used in this project adopt an **RAS+ coordinate convention** with an **identity affine** (no axis flips or rotations).

Accordingly:

* **x-axis (Jx):** Right–Left direction  
  * Positive values → **Right**
* **y-axis (Jy):** Anterior–Posterior direction  
  * Positive values → **Anterior**
* **z-axis (Jz):** Superior–Inferior direction  
  * Positive values → **Superior**

Under this convention, **Jx, Jy, and Jz represent the projections of the current density vector onto canonical anatomical axes**, enabling direct neuroanatomical interpretation of current flow direction.

---

### Voxel Selection Criteria

Computation is performed **only** for voxels satisfying:

* `_Jbrain.nii` ≠ 0  
* `|E|` ≠ 0  

All other voxels are set to **NaN**.

---

### Outputs (Subject Space)

```

<working_directory>/<subject_id>/
├── T1.nii
├── T1_tDCSLAB_Jbrain.nii
├── T1_tDCSLAB_Jbrain_x_fromEmag.nii
├── T1_tDCSLAB_Jbrain_y_fromEmag.nii
└── T1_tDCSLAB_Jbrain_z_fromEmag.nii

```

---

### MATLAB Implementations

```

batch_make_Jxyz_from_emag_and_Jbrain.m
batch_make_Jxyz_from_emag_and_Jbrain_SPECIAL_301428.m

```

* Direction is always taken from `T1_tDCSLAB_e.nii` (4D vector field)
* `_emag.nii` is **never** used for direction (magnitude-only)

---

### Notes

* Jx/Jy/Jz are derived **only in subject space**
* No spatial normalization is applied at this step
* Original ROAST outputs are **never modified**
* Supplementary processing exists solely to resolve data integrity issues

---

## Step 3: Spatial Normalization of Direction Maps to MNI152 Space (SPM12)

### Purpose

To enable **group-level, voxel-wise analysis**, masked **θ** and **φ** maps are normalized into **MNI152 space**.

---

### MATLAB Scripts

```

registration_all_auto_part_theta.m
registration_all_auto_part_phi.m

```

---

## Step 3b: Spatial Normalization of Current Density Magnitude and Components

### Purpose

To support group-level analysis of current density magnitude and Cartesian components, the following files are normalized to MNI152 space using deformation fields estimated from `T1.nii`:

* `_Jbrain.nii`
* `Jx`, `Jy`, `Jz`

---

### MATLAB Script

```

registration_all_auto_part_Jbrain_components.m

```

This script:

* Estimates deformation from `T1.nii`
* Applies the same deformation to:
  * `T1_tDCSLAB_Jbrain.nii`
  * `T1_tDCSLAB_Jbrain_x_fromEmag.nii`
  * `T1_tDCSLAB_Jbrain_y_fromEmag.nii`
  * `T1_tDCSLAB_Jbrain_z_fromEmag.nii`
* Produces warped outputs prefixed with `w`
* Does **not** process θ or φ maps

---

## E. End-to-End File Flow Summary

```

ROAST outputs (subject space)
↓
Step 1: Validate Jbrain
↓
Step 2: Compute θ / φ (subject space)
↓
Step 2b: Compute Jx / Jy / Jz (subject space)
↓
Step 3: Normalize θ / φ to MNI152
↓
Step 3b: Normalize Jbrain and Jx/Jy/Jz to MNI152

```

---

## F. Provenance and Reproducibility Notes

* Original ROAST files are **read-only**
* All derived files are written to the working directory
* Subject-space and MNI-space files coexist for auditability
* Coordinate conventions are explicitly documented
* Supplementary scripts exist only to resolve data integrity issues

---


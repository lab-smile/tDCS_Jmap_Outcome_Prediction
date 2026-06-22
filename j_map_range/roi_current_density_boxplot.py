"""
roi_current_density_boxplot.py

Plot violin + box + scatter (raincloud-style) figures showing
participant-level mean current-density magnitude within the top 7
predictive ROIs, separated by Responder vs Non-Responder groups.

Also computes and saves a shared value-range (vmin, vmax) per ROI
across all subjects, mirroring the compute_value_range logic from
BrainVisualizer, saved to: <OUTPUT_DIR>/roi_range/

Usage
-----
    python roi_current_density_boxplot.py
"""

from __future__ import annotations

# =============================================================================
# CONSTANTS — edit these as needed
# =============================================================================
JMAP_BASE_DIR = (
    "/blue/camctrp/working/junfu.cheng/roast_output_spm_registration_auto_accelerated/"
)
JMAP_MAGNITUDE_FILENAME = "wT1_tDCSLAB_Jbrain.nii"
JMAP_THETA_FILENAME     = "wT1_tDCSLAB_ThetaBrain.nii"
JMAP_PHI_FILENAME       = "wT1_tDCSLAB_PhiBrain.nii"

JMAP_XYZ_BASE_DIR = (
    "/orange/ruogu.fang/junfu.cheng/SMILE/j_map/j_map_direction/"
    "read_matlab_from_skylar/"
)
JMAP_X_FILENAME = "wT1_tDCSLAB_Jbrain_x_fromEmag.nii"
JMAP_Y_FILENAME = "wT1_tDCSLAB_Jbrain_y_fromEmag.nii"
JMAP_Z_FILENAME = "wT1_tDCSLAB_Jbrain_z_fromEmag.nii"

ATLAS_NAME = "cort-maxprob-thr25-2mm"

OUTPUT_DIR  = "figures/roi_boxplots"
OUTPUT_STEM = "top7_roi_current_density"
DPI         = 300

# Harvard–Oxford labels for the top-7 ROIs (must match atlas labels exactly)
TOP7_ROIS = [
    "Temporal Fusiform Cortex, posterior division",
    "Inferior Temporal Gyrus, posterior division",
    "Inferior Temporal Gyrus, temporooccipital part",
    "Supramarginal Gyrus, posterior division",
    "Lateral Occipital Cortex, inferior division",
    "Temporal Fusiform Cortex, anterior division",
    "Temporal Pole",
]

# Short labels for the figure x-axis (same order as TOP7_ROIS)
TOP7_SHORT_LABELS = [
    "TFC\n(post)",
    "ITG\n(post)",
    "ITG\n(temp-occ)",
    "SMG\n(post)",
    "LOC\n(inf)",
    "TFC\n(ant)",
    "Temporal\nPole",
]

# Colour palette
COLOR_RESPONDER     = "#2166AC"   # blue
COLOR_NONRESPONDER  = "#D6604D"   # red-orange

# =============================================================================
# Imports
# =============================================================================
import os
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import nibabel as nib
from nilearn import datasets, image

warnings.filterwarnings("ignore", category=FutureWarning)
matplotlib.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 9,
    }
)

# =============================================================================
# Hard-coded subject lists (from the problem statement)
# =============================================================================
TRAIN_SUBJECTS = {
    105256: 1, 105601: 1, 105971: 1, 107802: 1, 109021: 1,
    110081: 0, 115991: 0, 116036: 0, 202251: 0, 202808: 1,
    203846: 1,
}
TEST_SUBJECTS = {
    300700: 1, 301112: 0, 301428: 1, 301513: 0, 301538: 0,
    303009: 1, 303293: 0, 303346: 1, 303673: 0,
}


# =============================================================================
# Step 1 — Build subject DataFrame
# =============================================================================
def build_subject_df() -> pd.DataFrame:
    """Combine train and test subject dicts into a single DataFrame."""
    records = [
        {"subject_id": sid, "responder": label, "split": "train"}
        for sid, label in TRAIN_SUBJECTS.items()
    ] + [
        {"subject_id": sid, "responder": label, "split": "test"}
        for sid, label in TEST_SUBJECTS.items()
    ]
    df = pd.DataFrame(records).set_index("subject_id")
    return df


# =============================================================================
# Step 2 — Load valid NIfTI paths for magnitude maps
# =============================================================================
def load_valid_paths(
    subject_ids: List[int],
    base_dir: str,
    jmap_filename: str,
) -> Dict[int, str]:
    """
    Return a dict mapping subject_id -> nii_path for subjects whose
    file exists and loads without error.
    """
    valid: Dict[int, str] = {}
    for sid in subject_ids:
        sid_str  = str(sid)
        nii_path = os.path.join(base_dir, sid_str, jmap_filename)
        if not os.path.isdir(os.path.join(base_dir, sid_str)):
            print(f"[WARN] Missing folder for subject {sid}")
            continue
        if not os.path.isfile(nii_path):
            print(f"[WARN] Missing NIfTI for subject {sid}: {nii_path}")
            continue
        try:
            img = nib.load(nii_path)
            _   = img.get_fdata()          # validate readability
            valid[sid] = nii_path
        except Exception as exc:
            print(f"[ERROR] Corrupted NIfTI for subject {sid}: {exc}")
    return valid


# =============================================================================
# Step 3 — Build Harvard–Oxford ROI masks (resampled to atlas grid)
# =============================================================================
def build_roi_masks(
    roi_names: List[str],
    atlas_name: str = ATLAS_NAME,
) -> Tuple[Dict[str, np.ndarray], nib.Nifti1Image]:
    """
    Fetch the Harvard–Oxford cortical atlas and return binary masks for
    each requested ROI name.

    Returns
    -------
    masks : dict  roi_name -> bool ndarray (same grid as atlas)
    atlas_img : resampled atlas NIfTI (used as the reference grid)
    """
    ho     = datasets.fetch_atlas_harvard_oxford(atlas_name)
    labels = [
        l.decode("utf-8") if isinstance(l, (bytes, bytearray)) else str(l)
        for l in ho.labels
    ]
    maps   = ho.maps
    atlas_img: nib.Nifti1Image = (
        maps if isinstance(maps, nib.spatialimages.SpatialImage)
        else nib.load(maps)
    )
    atlas_arr = atlas_img.get_fdata().astype(int)

    masks: Dict[str, np.ndarray] = {}
    for roi in roi_names:
        q = roi.strip().casefold()
        # Exact match first, then substring
        indices = [i for i, n in enumerate(labels) if n.casefold() == q and i != 0]
        if not indices:
            indices = [i for i, n in enumerate(labels) if q in n.casefold() and i != 0]
        if not indices:
            print(f"[WARN] ROI not found in atlas: '{roi}' — skipping.")
            continue
        if len(indices) > 1:
            print(
                f"[WARN] ROI '{roi}' matched {len(indices)} labels "
                f"({[labels[i] for i in indices]}); using all of them."
            )
        mask = np.isin(atlas_arr, indices)
        if not mask.any():
            print(f"[WARN] Empty mask for ROI '{roi}' — skipping.")
            continue
        masks[roi] = mask
        print(
            f"[INFO] ROI '{roi}': {mask.sum()} voxels "
            f"(index/indices {indices})"
        )
    return masks, atlas_img


# =============================================================================
# Step 4 — Compute per-subject, per-ROI mean current-density magnitude
#           AND compute the shared value range across ALL subjects per ROI
# =============================================================================
def compute_roi_means_and_range(
    subject_paths: Dict[int, str],
    roi_masks:     Dict[str, np.ndarray],
    atlas_img:     nib.Nifti1Image,
    clip_negative: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Tuple[float, float]]]:
    """
    For each subject × ROI:
      - Resample subject image to the atlas grid
      - Extract voxel values within the ROI mask
      - Optionally clip negative values to 0  (current density magnitude >= 0)
      - Record the mean

    Simultaneously, track the global vmin/vmax across ALL subjects for each
    ROI (mirrors BrainVisualizer.compute_value_range logic).

    Parameters
    ----------
    subject_paths : dict  subject_id -> nii_path
    roi_masks     : dict  roi_name   -> bool ndarray (atlas grid)
    atlas_img     : NIfTI used as the resampling target
    clip_negative : bool
        If True, voxel values < 0 are set to 0 before computing statistics.
        Physically correct for current-density magnitude (J ≥ 0).
        Mirrors the old-code block:
            if value_range[0] < 0:
                value_range = (0, value_range[1])

    Returns
    -------
    df_means  : DataFrame  index=subject_id, columns=roi_names, values=mean [A/m²]
    roi_range : dict  roi_name -> (vmin, vmax)  — shared range across all subjects
                Used to keep colormaps consistent across participants,
                exactly as compute_value_range() provided in the old code.
    """
    # Accumulate all per-subject voxel values per ROI so we can compute the
    # true global min/max AFTER processing every subject (same logic as
    # compute_value_range which iterates X_all before plotting anything).
    roi_all_voxels: Dict[str, List[np.ndarray]] = {roi: [] for roi in roi_masks}

    records: List[Dict] = []

    for sid, nii_path in subject_paths.items():
        row: Dict = {"subject_id": sid}
        try:
            subj_img = nib.load(nii_path)
            # Resample subject image → atlas grid (nearest for label maps,
            # continuous for functional/metric data such as current density)
            subj_res = image.resample_to_img(
                subj_img,
                atlas_img,
                interpolation="continuous",
                force_resample=True,
                copy_header=True,
            )
            subj_data = subj_res.get_fdata()
        except Exception as exc:
            print(f"[ERROR] Could not process subject {sid}: {exc}")
            for roi in roi_masks:
                row[roi] = np.nan
            records.append(row)
            continue

        for roi, mask in roi_masks.items():
            if subj_data.shape[:3] != mask.shape:
                print(
                    f"[WARN] Shape mismatch for subject {sid}, ROI '{roi}': "
                    f"{subj_data.shape} vs mask {mask.shape}. Skipping ROI."
                )
                row[roi] = np.nan
                continue

            voxels = subj_data[mask]

            # Keep only finite values
            voxels = voxels[np.isfinite(voxels)]

            # --- Clip negatives to 0 (current density magnitude is non-negative)
            # This mirrors the old-code logic:
            #   if value_range[0] < 0:
            #       value_range = (0, value_range[1])
            # but applied at the voxel level so means are also non-negative.
            if clip_negative:
                voxels = np.clip(voxels, 0.0, None)

            # Accumulate for global range computation
            if len(voxels) > 0:
                roi_all_voxels[roi].append(voxels)

            row[roi] = float(np.mean(voxels)) if len(voxels) > 0 else np.nan

        records.append(row)
        print(f"[INFO] Subject {sid} done.")

    df_means = pd.DataFrame(records).set_index("subject_id")

    # --- Compute shared value range across ALL subjects per ROI ---
    # This is the direct equivalent of:
    #   value_range = viz.compute_value_range(X_all, roi_name, ref_img)
    # The range is computed over the union of all subjects' ROI voxels so
    # that any subsequent per-subject visualization uses an identical scale.
    roi_range: Dict[str, Tuple[float, float]] = {}
    for roi, voxel_list in roi_all_voxels.items():
        if not voxel_list:
            print(f"[WARN] No valid voxels collected for ROI '{roi}'; range set to (0, 1).")
            roi_range[roi] = (0.0, 1.0)
            continue
        all_vals = np.concatenate(voxel_list)
        vmin = float(np.nanmin(all_vals))
        vmax = float(np.nanmax(all_vals))
        # Guard against degenerate (flat) distributions
        if vmax == vmin:
            vmax = vmin + 1e-6
        roi_range[roi] = (vmin, vmax)
        print(f"[INFO] ROI '{roi}': value range = ({vmin:.4e}, {vmax:.4e})")

    return df_means, roi_range


# =============================================================================
# Step 5 — Save the ROI value ranges to disk
# =============================================================================
def save_roi_range(
    roi_range: Dict[str, Tuple[float, float]],
    out_dir:   str,
    stem:      str = "roi_value_range",
) -> Dict[str, str]:
    """
    Save the per-ROI (vmin, vmax) table to CSV and a human-readable TXT.

    Output folder: <out_dir>/roi_range/

    Returns
    -------
    dict of saved file paths.
    """
    range_dir = os.path.join(out_dir, "roi_range")
    os.makedirs(range_dir, exist_ok=True)

    rows = [
        {"roi_name": roi, "vmin": vmin, "vmax": vmax}
        for roi, (vmin, vmax) in roi_range.items()
    ]
    df_range = pd.DataFrame(rows)

    csv_path = os.path.join(range_dir, f"{stem}.csv")
    df_range.to_csv(csv_path, index=False)

    txt_path = os.path.join(range_dir, f"{stem}.txt")
    with open(txt_path, "w") as fh:
        fh.write("Per-ROI shared value range (vmin, vmax) across all subjects\n")
        fh.write("=" * 64 + "\n")
        fh.write(f"{'ROI':<55}  {'vmin':>12}  {'vmax':>12}\n")
        fh.write("-" * 64 + "\n")
        for roi, (vmin, vmax) in roi_range.items():
            fh.write(f"{roi:<55}  {vmin:>12.4e}  {vmax:>12.4e}\n")

    print(f"[INFO] ROI value ranges saved → {csv_path}")
    print(f"[INFO] ROI value ranges saved → {txt_path}")
    return {"csv": csv_path, "txt": txt_path}


# =============================================================================
# Step 6 — Plotting
# =============================================================================
def plot_roi_comparison(
    df_means:    pd.DataFrame,
    df_subjects: pd.DataFrame,
    roi_names:   List[str],
    short_labels: List[str],
    out_dir:     str = OUTPUT_DIR,
    out_stem:    str = OUTPUT_STEM,
    dpi:         int = DPI,
) -> Dict[str, str]:
    """
    For each of the 7 ROIs, draw a half-violin + box + strip (beeswarm)
    plot comparing Responders vs Non-Responders.

    Layout: 1 row × 7 columns (one subplot per ROI).

    Returns
    -------
    dict of saved file paths.
    """
    # ---- merge group labels ----
    df_plot = df_means.join(df_subjects[["responder"]], how="inner")
    df_plot["Group"] = df_plot["responder"].map({1: "Responder", 0: "Non-Responder"})

    group_order  = ["Responder", "Non-Responder"]
    group_colors = {"Responder": COLOR_RESPONDER, "Non-Responder": COLOR_NONRESPONDER}
    x_positions  = {g: i for i, g in enumerate(group_order)}   # 0 or 1

    n_rois = len(roi_names)
    fig, axes = plt.subplots(
        1, n_rois,
        figsize=(2.6 * n_rois, 5.5),
        sharey=False,
    )
    if n_rois == 1:
        axes = [axes]

    for ax, roi, short_lbl in zip(axes, roi_names, short_labels):
        if roi not in df_plot.columns:
            ax.set_visible(False)
            continue

        col_data = df_plot[["Group", roi]].dropna(subset=[roi])

        for group in group_order:
            gdata  = col_data.loc[col_data["Group"] == group, roi].values
            xpos   = x_positions[group]
            color  = group_colors[group]

            if len(gdata) == 0:
                continue

            # --- half violin (kernel density estimate) ---
            if len(gdata) >= 3:
                try:
                    from scipy.stats import gaussian_kde
                    kde     = gaussian_kde(gdata, bw_method=0.4)
                    y_grid  = np.linspace(gdata.min(), gdata.max(), 200)
                    density = kde(y_grid)
                    density = density / density.max() * 0.38   # half-width
                    side    = 1 if xpos == 0 else -1           # mirror for right group
                    ax.fill_betweenx(
                        y_grid,
                        xpos,
                        xpos + side * density,
                        color=color,
                        alpha=0.35,
                        linewidth=0,
                    )
                except Exception:
                    pass   # fall back gracefully if KDE fails

            # --- box plot (manual: IQR box + whiskers + median) ---
            q1, med, q3 = np.percentile(gdata, [25, 50, 75])
            iqr  = q3 - q1
            lo   = max(gdata.min(), q1 - 1.5 * iqr)
            hi   = min(gdata.max(), q3 + 1.5 * iqr)
            bw   = 0.12   # half box-width
            rect = mpatches.FancyBboxPatch(
                (xpos - bw, q1), 2 * bw, iqr,
                boxstyle="square,pad=0",
                linewidth=1.4,
                edgecolor=color,
                facecolor=color,
                alpha=0.55,
            )
            ax.add_patch(rect)
            # median line
            ax.hlines(med, xpos - bw, xpos + bw, colors=color, linewidth=2.2, zorder=5)
            # whiskers
            ax.vlines(xpos, lo, q1, colors=color, linewidth=1.2, linestyle="--")
            ax.vlines(xpos, q3, hi, colors=color, linewidth=1.2, linestyle="--")
            # whisker caps
            cap_w = bw * 0.5
            ax.hlines(lo, xpos - cap_w, xpos + cap_w, colors=color, linewidth=1.2)
            ax.hlines(hi, xpos - cap_w, xpos + cap_w, colors=color, linewidth=1.2)

            # --- scatter jitter ---
            rng    = np.random.default_rng(seed=42 + xpos)
            jitter = rng.uniform(-0.07, 0.07, size=len(gdata))
            ax.scatter(
                xpos + jitter,
                gdata,
                color=color,
                s=28,
                alpha=0.80,
                zorder=6,
                edgecolors="white",
                linewidths=0.4,
            )

        # axes formatting
        ax.set_xlim(-0.6, 1.6)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Resp.", "Non-\nResp."], fontsize=7.5)
        ax.set_title(short_lbl, fontsize=8.5, pad=4)
        ax.set_xlabel("")
        if ax == axes[0]:
            ax.set_ylabel("Mean current density (A/m²)", fontsize=9)
        else:
            ax.set_ylabel("")

        ax.yaxis.grid(True, linestyle=":", linewidth=0.6, alpha=0.6)
        ax.set_axisbelow(True)

    # ---- shared legend ----
    legend_handles = [
        mpatches.Patch(color=COLOR_RESPONDER,    label="Responder"),
        mpatches.Patch(color=COLOR_NONRESPONDER, label="Non-Responder"),
        Line2D([0], [0], color="gray", linewidth=1.2,
               linestyle="--", label="Whiskers (1.5×IQR)"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=3,
        fontsize=8.5,
        frameon=False,
        bbox_to_anchor=(0.5, -0.04),
    )
    fig.suptitle(
        "Mean Current-Density Magnitude in Top-7 Predictive ROIs\n"
        "Responders vs Non-Responders",
        fontsize=11,
        y=1.02,
    )
    plt.tight_layout(rect=[0, 0.04, 1, 1])

    # ---- save ----
    os.makedirs(out_dir, exist_ok=True)
    saved: Dict[str, str] = {}
    for ext in ("png", "svg", "pdf"):
        path = os.path.join(out_dir, f"{out_stem}.{ext}")
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        saved[ext] = path
        print(f"[INFO] Saved → {path}")

    plt.show()
    plt.close(fig)
    return saved


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    # 1. Subject metadata
    df_subjects = build_subject_df()
    all_sids    = df_subjects.index.tolist()

    # 2. Validate NIfTI paths (magnitude map)
    print("\n--- Scanning for magnitude NIfTI files ---")
    subject_paths = load_valid_paths(all_sids, JMAP_BASE_DIR, JMAP_MAGNITUDE_FILENAME)
    print(f"[INFO] {len(subject_paths)}/{len(all_sids)} subjects have valid magnitude maps.\n")

    if not subject_paths:
        raise RuntimeError("No valid subject NIfTI files found. Check JMAP_BASE_DIR.")

    # 3. Build ROI masks
    print("--- Building Harvard–Oxford ROI masks ---")
    roi_masks, atlas_img = build_roi_masks(TOP7_ROIS, atlas_name=ATLAS_NAME)
    if not roi_masks:
        raise RuntimeError("No ROI masks could be built. Check TOP7_ROIS and atlas.")

    # 4. Compute per-subject ROI means AND shared value ranges
    #    (replicates the BrainVisualizer.compute_value_range(X_all, ...) logic,
    #     including the clip-negative-to-zero correction from the old code block)
    print("\n--- Computing per-subject ROI means and shared value ranges ---")
    df_means, roi_range = compute_roi_means_and_range(
        subject_paths,
        roi_masks,
        atlas_img,
        clip_negative=True,   # J magnitude >= 0 by definition
    )
    print(f"\nROI mean summary:\n{df_means.describe().round(6)}\n")

    # 5. Save the shared value ranges to roi_range/
    print("--- Saving ROI value ranges ---")
    range_paths = save_roi_range(roi_range, out_dir=OUTPUT_DIR)

    # 6. Plot
    print("--- Plotting ---")
    saved = plot_roi_comparison(
        df_means     = df_means,
        df_subjects  = df_subjects,
        roi_names    = TOP7_ROIS,
        short_labels = TOP7_SHORT_LABELS,
        out_dir      = OUTPUT_DIR,
        out_stem     = OUTPUT_STEM,
        dpi          = DPI,
    )
    print("\nDone. Saved files:")
    for ext, path in saved.items():
        print(f"  {ext}: {path}")
    print("\nROI range files:")
    for ext, path in range_paths.items():
        print(f"  {ext}: {path}")


if __name__ == "__main__":
    main()
# -*- coding: utf-8 -*-
from __future__ import annotations

import ast
import os
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple, Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colormaps, colors

import nibabel as nib
from nilearn import datasets, image, plotting
from nilearn.image import new_img_like


# ----------------------------- Data Model ----------------------------- #

@dataclass(frozen=True)
class PlotConfig:
    metric_name: str = "SHAP"
    background: str = "black"
    top_n_regions: Optional[int] = None
    output_dir: str = "figure_brain"
    heatmap_filename: str = "roi_shap_heatmap.png"
    bar_filename_prefix: str = "roi_contrib"
    atlas_name: str = "cort-maxprob-thr25-2mm"          # cortical
    subcortical_atlas_name: str = "sub-maxprob-thr25-2mm"  # subcortical
    include_subcortical: bool = True
    display_mode: str = "ortho"
    cut_coords: Optional[Tuple[float, float, float]] = None


@dataclass(frozen=True)
class PlotOutputs:
    roi_df: pd.DataFrame
    bar_path: str
    heatmap_path: str


# ----------------------------- I/O Helpers ---------------------------- #

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_reference_image(reference: str | nib.spatialimages.SpatialImage) -> nib.spatialimages.SpatialImage:
    """Load a reference NIfTI image (grid) from a path or pass-through an already loaded image."""
    if isinstance(reference, str):
        return nib.load(reference)
    return reference


def read_shap_csv(shap_csv_path: str) -> pd.DataFrame:
    """Read SHAP CSV and validate required columns."""
    df = pd.read_csv(shap_csv_path)
    req_cols = {"voxel_ijk", "mean_abs_shap_voxel"}
    missing = [c for c in req_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{shap_csv_path!r} missing required column(s): {missing}")
    return df


# ----------------------------- Parsing -------------------------------- #

def parse_voxel_ijk(value) -> Tuple[int, int, int]:
    """
    Robustly parse a voxel IJK triple from various possible CSV encodings:
    - tuple/list/array of length ≥3
    - string that can be literal_eval'ed into a tuple/list
    - string like '(i, j, k)' or 'i, j, k'
    """
    if isinstance(value, (tuple, list, np.ndarray)) and len(value) >= 3:
        return int(value[0]), int(value[1]), int(value[2])

    if isinstance(value, str):
        # Try literal_eval first
        try:
            tup = ast.literal_eval(value)
            return int(tup[0]), int(tup[1]), int(tup[2])
        except Exception:
            # Fallback: strip and split on commas
            s = value.strip().strip("()[]")
            parts = [p.strip() for p in s.split(",")]
            if len(parts) >= 3:
                return int(parts[0]), int(parts[1]), int(parts[2])

    raise ValueError(f"Cannot parse voxel_ijk entry: {value!r}")


# ----------------------------- Volume Builders ------------------------ #

def build_sparse_abs_volume(
    ijk_list: Sequence[Tuple[int, int, int]],
    values: np.ndarray,
    reference_img: nib.spatialimages.SpatialImage,
) -> nib.spatialimages.SpatialImage:
    """Create a sparse |SHAP| 3D volume aligned to the reference grid."""
    ref_shape = reference_img.shape
    out = np.full(ref_shape, np.nan, dtype=float)

    for (i, j, k), v in zip(ijk_list, values):
        if 0 <= i < ref_shape[0] and 0 <= j < ref_shape[1] and 0 <= k < ref_shape[2]:
            out[i, j, k] = float(v)

    return new_img_like(reference_img, out)

def fetch_and_combine_harvard_oxford(
    reference_img: nib.spatialimages.SpatialImage,
    cortical_name: str = "cort-maxprob-thr25-2mm",
    subcortical_name: str = "sub-maxprob-thr25-2mm",
) -> Tuple[nib.spatialimages.SpatialImage, np.ndarray, List[str]]:

    # --- Cortical ---
    c = datasets.fetch_atlas_harvard_oxford(cortical_name)
    c_img = image.resample_to_img(c.maps, reference_img, interpolation="nearest",
                                  force_resample=True, copy_header=True)
    try:
        c_arr = image.get_fdata(c_img)
    except Exception:
        c_arr = image.get_data(c_img)
    c_arr = np.asarray(c_arr, dtype=int)
    c_labels = list(c.labels)  # 0=Background
    Nc = len(c_labels)

    # --- Subcortical ---
    s = datasets.fetch_atlas_harvard_oxford(subcortical_name)
    s_img = image.resample_to_img(s.maps, reference_img, interpolation="nearest",
                                  force_resample=True, copy_header=True)
    try:
        s_arr = image.get_fdata(s_img)
    except Exception:
        s_arr = image.get_data(s_img)
    s_arr = np.asarray(s_arr, dtype=int)
    s_labels = list(s.labels)  # 0=Background
    Ns = len(s_labels)

    # --- Combine with *cortical precedence* ---
    combined = np.zeros_like(c_arr, dtype=int)

    # keep cortical wherever it’s defined
    cortical_mask = c_arr > 0
    combined[cortical_mask] = c_arr[cortical_mask]

    # fill only where cortex is background
    empty_mask = ~cortical_mask
    for j in range(1, Ns):
        new_idx = (Nc - 1) + j
        mask_j = (s_arr == j) & empty_mask
        combined[mask_j] = new_idx

    combined_labels = ["Background"]
    combined_labels += [f"Cortex: {n}" for n in c_labels[1:]]
    combined_labels += [f"Subcortex: {n}" for n in s_labels[1:]]


    combined_img = new_img_like(reference_img, combined)
    return combined_img, combined, combined_labels


def print_labels(combined_labels):
    for index, label in enumerate(combined_labels):
        print(f"{index}: ", label)

def fetch_resampled_harvard_oxford(
    reference_img: nib.spatialimages.SpatialImage,
    atlas_name: str = "cort-maxprob-thr25-2mm",
) -> Tuple[nib.spatialimages.SpatialImage, np.ndarray, List[str]]:
    """
    Fetch Harvard–Oxford atlas and resample to the reference grid.
    Returns (atlas_img, atlas_arr, atlas_labels).
    """
    atlas = datasets.fetch_atlas_harvard_oxford(atlas_name)
    atlas_img = image.resample_to_img(
        atlas.maps, reference_img, interpolation="nearest", force_resample=True, copy_header=True
    )
    # get_fdata preferred; fallback to deprecated get_data if needed
    try:
        atlas_arr = image.get_fdata(atlas_img)
    except Exception:
        atlas_arr = image.get_data(atlas_img)
    atlas_arr = np.asarray(atlas_arr, dtype=int)
    atlas_labels = list(atlas.labels)  # 0 = Background
    return atlas_img, atlas_arr, atlas_labels


# ----------------------------- ROI Math -------------------------------- #

def compute_mean_abs_by_roi(
    abs_volume_img: nib.spatialimages.SpatialImage,
    atlas_arr: np.ndarray,
    atlas_labels: Sequence[str],
) -> pd.DataFrame:
    """
    Compute mean(|SHAP|) per ROI, excluding background/unlabeled ROIs.
    Returns a DataFrame with columns: roi_index, roi_name, mean_abs_shap
    """
    
    # Load the |SHAP| array
    try:
        abs_arr = image.get_fdata(abs_volume_img)
    except Exception:
        abs_arr = image.get_data(abs_volume_img)
    abs_arr = abs_arr.astype(float)
    print("print(abs_arr.shape)")
    print(abs_arr.shape)
    path_nii = os.path.join("figure_brain", "abs_arr_visualize.nii.gz")
    path_hist = os.path.join("figure_brain", "abs_arr_hist.png")
    save_nifti_and_hist(abs_arr, path_nii, path_hist)

    rows: List[Tuple[int, str, float]] = []
    for roi_idx in range(1, len(atlas_labels)):
        roi_name = str(atlas_labels[roi_idx])
        if roi_name.lower() in {"background", "unknown", "unlabeled"}:
            continue

        mask = (atlas_arr == roi_idx)
        if not np.any(mask):
            continue

        vals = abs_arr[mask]
        # if np.all(np.isnan(vals)):
        #     continue

        rows.append((roi_idx, roi_name, float(np.nanmean(vals))))

    if not rows:
        raise ValueError("No valid ROIs found with |SHAP| values.")

    df = pd.DataFrame(rows, columns=["roi_index", "roi_name", "mean_abs_shap"])
    df.sort_values("mean_abs_shap", ascending=False, inplace=True, ignore_index=True)
    
    value_col="mean_abs_shap"
    #n, method, top_df = choose_top_n_by_shap(df,value_col)
    #print("n, method, top_df:", n,',', method,',', top_df)
    
    return df


def choose_top_n_by_shap(
    df,
    value_col="mean_abs_shap",
    min_n=5,
    pareto_threshold=0.33,
    drop_threshold=0.35,
    knee_max_fraction=0.5,
):
    """
    Automatically choose n for 'top n' based on the distribution of a value column.

    Strategy:
    1. Sort values descending.
    2. Try to find a clear 'elbow' (big relative drop). If found early enough, use it.
    3. Otherwise, use Pareto: smallest n explaining `pareto_threshold` of total mass.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing the values.
    value_col : str
        Name of the column with importance scores (e.g. mean_abs_shap).
    min_n : int
        Minimum number of items to keep.
    pareto_threshold : float
        Fraction of total sum to keep in Pareto method (e.g. 0.8 = 80%).
    drop_threshold : float
        Minimum relative drop between consecutive values to accept an 'elbow'.
        Example: 0.35 means a 35% drop.
    knee_max_fraction : float
        Ignore elbows that occur too late (e.g. after 50% of the list).
    
    Returns
    -------
    n : int
        Chosen number of items.
    method : str
        "elbow" or "pareto".
    top_df : pd.DataFrame
        Dataframe with the top n rows.
    """
    # 1. Clean and sort
    df_clean = df.copy()
    df_clean = df_clean.dropna(subset=[value_col])
    df_clean = df_clean.sort_values(value_col, ascending=False)

    values = df_clean[value_col].to_numpy()
    m = len(values)

    if m == 0:
        return 0, "none", df_clean.head(0)
    if m <= min_n:
        return m, "all", df_clean

    # 2. Try to find an elbow based on relative drop
    #    rel_drop[i] = (v[i] - v[i+1]) / v[i]
    v1 = values[:-1]
    v2 = values[1:]
    rel_drop = (v1 - v2) / np.maximum(v1, 1e-30)   # avoid divide-by-zero

    elbow_idx = int(np.argmax(rel_drop))   # index of largest relative drop between i and i+1
    elbow_drop = rel_drop[elbow_idx]
    elbow_n = elbow_idx + 1                # number of items if we cut at this elbow

    # Check if the elbow is 'early enough' and strong enough
    if (
        elbow_drop >= drop_threshold
        and elbow_n >= min_n
        and elbow_n <= int(knee_max_fraction * m)
    ):
        n = elbow_n
        method = "elbow"
    else:
        # 3. Fall back to Pareto (cumulative contribution)
        total = values.sum()
        if total <= 0:
            # Degenerate case: all zeros or negative (unlikely here)
            n = min_n
            method = "fallback_min"
        else:
            cum_frac = np.cumsum(values) / total
            # smallest n such that cum_frac >= pareto_threshold
            n = int(np.searchsorted(cum_frac, pareto_threshold) + 1)
            n = max(n, min_n)
            method = "pareto"

    # Clip n to available rows
    n = min(n, m)
    top_df = df_clean.head(n)
    return n, method, top_df


def save_numpy_as_nifti(np_array, output_path, voxel_sizes=(1.0, 1.0, 1.0)):
    """
    Save a NumPy array as a compressed NIfTI (.nii.gz) file with assigned voxel sizes.

    Parameters
    ----------
    np_array : np.ndarray
        The array to save (e.g., shape (99,117,95)).
    output_path : str
        File path where the .nii.gz file will be written.
    voxel_sizes : tuple of float
        (x, y, z) voxel dimensions in mm.
    """

    # Ensure the filename ends with .nii.gz
    if not output_path.endswith(".nii.gz"):
        output_path = output_path.rstrip(".nii") + ".nii.gz"

    # Unpack voxel sizes
    vx, vy, vz = voxel_sizes

    # Construct a simple affine that encodes voxel spacing
    affine = np.array([
        [vx, 0,  0,  0],
        [0,  vy, 0,  0],
        [0,  0,  vz, 0],
        [0,  0,  0,  1]
    ])

    # Create NIfTI image
    nifti_img = nib.Nifti1Image(np_array, affine)

    # Save file
    nib.save(nifti_img, output_path)
    print(f"Saved compressed NIfTI file to {output_path}")
    
def save_nifti_and_hist(np_array, nifti_path, hist_path, affine=None, bins=100):
    """
    Save a NumPy array as a NIfTI file and save histogram of voxel values.
    
    Parameters
    ----------
    np_array : np.ndarray
        Array to save, e.g., shape (99, 117, 95)
    nifti_path : str
        Output path for .nii or .nii.gz file
    hist_path : str
        Output path for histogram image (.png, .jpg, etc.)
    affine : np.ndarray, optional
        4x4 affine matrix. If None, uses identity.
    bins : int
        Number of histogram bins
    """

    # ------------------------------------
    # Save as NIfTI
    # ------------------------------------
    if affine is None:
        affine = np.eye(4)

    nifti_img = nib.Nifti1Image(np_array, affine)
    nib.save(nifti_img, nifti_path)
    print(f"Saved NIfTI to: {nifti_path}")

    # ------------------------------------
    # Plot and save histogram
    # ------------------------------------
    data_flat = np_array.flatten()

    plt.figure(figsize=(8, 6))
    plt.hist(data_flat, bins=bins, edgecolor='black')
    plt.title("Voxel Intensity Histogram")
    plt.xlabel("Intensity")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(hist_path, dpi=150)
    plt.close()

    print(f"Saved histogram to: {hist_path}")


# ----------------------------- Color & Normalization ------------------- #

def make_shared_normalizer(values: Iterable[float]) -> Tuple[colors.Normalize, float, float]:
    vals = np.asarray(list(values), dtype=float)
    vmin = float(np.nanmin(vals))
    vmax = float(np.nanmax(vals))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax == vmin:
        vmax = vmin + 1e-6
    return colors.Normalize(vmin=vmin, vmax=vmax), vmin, vmax


def pick_background_colors(background: str) -> Tuple[str, str]:
    bg = "black" if str(background).lower() == "black" else "white"
    fg = "white" if bg == "black" else "black"
    return bg, fg


# ----------------------------- Plotting -------------------------------- #

def plot_roi_bar_chart(
    roi_df: pd.DataFrame,
    cfg: PlotConfig,
    norm: colors.Normalize,
    cmap_name: str = "coolwarm",
) -> str:
    """Save a horizontal bar chart of ROI mean(|SHAP|) with colors tied to heatmap scale."""
    _ensure_dir(cfg.output_dir)
    bg_color, text_color = pick_background_colors(cfg.background)

    data = roi_df.copy()
    if cfg.top_n_regions is not None:
        data = data.head(int(cfg.top_n_regions)).copy()

    cmap = colormaps.get_cmap(cmap_name)
    bar_colors = [cmap(norm(v)) for v in data["mean_abs_shap"][::-1]]

    fig, ax = plt.subplots(
        figsize=(10, max(4, 0.35 * len(data))),
        facecolor=bg_color
    )
    ax.barh(
        data["roi_name"][::-1],
        data["mean_abs_shap"][::-1],
        color=bar_colors,
        edgecolor=text_color,
        linewidth=0.5
    )
    #ax.set_xlabel("Mean |SHAP| per ROI", color=text_color)
    ax.set_xlabel("ROI-Level Spatial Density of SHAP Importance", color=text_color)
    ax.set_title(f"Mean ROI Contributions ({cfg.metric_name}) — Harvard–Oxford", color=text_color)
    ax.tick_params(colors=text_color)
    for spine in ax.spines.values():
        spine.set_color(text_color)
    fig.tight_layout()

    slug = cfg.metric_name.replace(" ", "_").lower()
    bar_path = os.path.join(cfg.output_dir, f"{cfg.bar_filename_prefix}_{slug}_bar.png")
    fig.savefig(bar_path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    return bar_path


def build_roi_mean_image(
    roi_df: pd.DataFrame,
    atlas_img: nib.spatialimages.SpatialImage,
    atlas_arr: np.ndarray,
    vmin: float,
    vmax: float,
    top_n: Optional[int] = None
) -> nib.spatialimages.SpatialImage:
    """
    Build a per-voxel mean(|SHAP|) ROI map so the heatmap uses ROI means
    and matches the bar chart color scale.
    
    If top_n is specified, only the top N ROIs are included in the map.
    """
    # Sentinel value below vmin that will render transparent via "under"
    sentinel = vmin - (vmax - vmin + np.finfo(float).eps)
    mean_map = np.full(atlas_arr.shape, sentinel, dtype=float)

    # Filter to top N if specified
    data = roi_df.head(top_n) if top_n is not None else roi_df
    
    mean_by_idx: Dict[int, float] = dict(zip(data["roi_index"], data["mean_abs_shap"]))
    for idx, mval in mean_by_idx.items():
        mean_map[atlas_arr == idx] = float(mval)

    return new_img_like(atlas_img, mean_map)


def plot_roi_heatmap(
    mean_img: nib.spatialimages.SpatialImage,
    cfg: PlotConfig,
    vmin: float,
    vmax: float,
    cmap_name: str = "coolwarm",
) -> str:
    """Save a large, tight ROI heatmap using the same color scale as the bar chart."""
    _ensure_dir(cfg.output_dir)
    bg_color, _ = pick_background_colors(cfg.background)
    black_bg = (bg_color == "black")

    # Transparent "under" so non-ROI voxels don't show
    cmap = colormaps.get_cmap(cmap_name).with_extremes(under=(0, 0, 0, 0))

    # High-res anatomical background for aesthetics
    bg_img = datasets.load_mni152_template(resolution=1)

    fig = plt.figure(figsize=(12, 7), facecolor=bg_color)
    brain_ax = fig.add_axes([0.04, 0.06, 0.80, 0.88])  # large central panel

    title_suffix = f" (Top {cfg.top_n_regions})" if cfg.top_n_regions else ""
    disp = plotting.plot_img(
        mean_img,
        bg_img=bg_img,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        colorbar=True,
        figure=fig,
        axes=brain_ax,
        black_bg=black_bg,
        title=f"Mean |SHAP| per ROI (Harvard–Oxford){title_suffix}",
        display_mode=cfg.display_mode,
        cut_coords=cfg.cut_coords,
    )

    # Make the colorbar thin and centered on the right
    cbar_ax = fig.axes[-1]
    cbar_ax.set_position([0.87, 0.25, 0.025, 0.5])

    out_path = os.path.join(cfg.output_dir, cfg.heatmap_filename)
    disp.savefig(out_path, dpi=600, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return out_path


# ----------------------------- Orchestrator --------------------------- #

def plot_shap_from_csv_modular(
    shap_csv_path: str,
    reference_img: str | nib.spatialimages.SpatialImage,
    *,
    metric_name: str = "SHAP",
    background: str = "black",
    top_n_regions: Optional[int] = None,
    output_dir: str = "figure_brain",
    atlas_name: str = "cort-maxprob-thr25-2mm",
    subcortical_atlas_name: str = "sub-maxprob-thr25-2mm",
    include_subcortical: bool = True,
    display_mode: str = "ortho",
    cut_coords: Optional[Tuple[float, float, float]] = None,
    quiet: bool = True,
) -> PlotOutputs:
    """
    Modular pipeline:
      1) Load grid & SHAP CSV
      2) Build sparse |SHAP| volume
      3) Fetch & resample Harvard–Oxford atlas
      4) Compute ROI means of |SHAP|
      5) Plot bar chart (colors = shared scale, showing top N)
      6) Build ROI-mean image (only top N) & plot heatmap

    Returns PlotOutputs(roi_df, bar_path, heatmap_path).
    """
    cfg = PlotConfig(
        metric_name=metric_name,
        background=background,
        top_n_regions=top_n_regions,
        output_dir=output_dir,
        atlas_name=atlas_name,
        subcortical_atlas_name=subcortical_atlas_name,
        include_subcortical=include_subcortical,
        display_mode=display_mode,
        cut_coords=cut_coords,
    )
    _ensure_dir(cfg.output_dir)

    # 1) Load reference & SHAP CSV
    ref_img = load_reference_image(reference_img)
    df = read_shap_csv(shap_csv_path)

    # 2) Sparse |SHAP| volume
    ijk_list = [parse_voxel_ijk(v) for v in df["voxel_ijk"]]
    shap_vals = df["mean_abs_shap_voxel"].to_numpy(dtype=float)
    shap_abs_img = build_sparse_abs_volume(ijk_list, shap_vals, ref_img)

    # 3) Atlas (aligned to grid): cortical-only or cortical+subcortical
    if include_subcortical and subcortical_atlas_name:
        atlas_img, atlas_arr, atlas_labels = fetch_and_combine_harvard_oxford(
            ref_img, atlas_name, subcortical_atlas_name
        )
        
    else:
        atlas_img, atlas_arr, atlas_labels = fetch_resampled_harvard_oxford(
            ref_img, atlas_name
        )

    # 4) Mean(|SHAP|) per ROI
    roi_df = compute_mean_abs_by_roi(shap_abs_img, atlas_arr, atlas_labels)
    roi_df_path = os.path.join(output_dir, "roi_contrib_shap_mean.csv")
    roi_df.to_csv(roi_df_path)
    if not quiet:
        print(f"[OK] Saved table to: {roi_df_path}")
        

    # 5) Determine which ROIs to display (top N or all)
    display_df = roi_df.head(cfg.top_n_regions) if cfg.top_n_regions else roi_df
    
    # 5a) Shared color normalization based on displayed ROIs only
    norm, vmin, vmax = make_shared_normalizer(display_df["mean_abs_shap"])
    
    # 5b) Plot bar chart with top N
    bar_path = plot_roi_bar_chart(roi_df, cfg, norm)

    # 6) ROI mean image (only top N) and heatmap
    mean_img = build_roi_mean_image(roi_df, atlas_img, atlas_arr, vmin, vmax, top_n=cfg.top_n_regions)
    heatmap_path = plot_roi_heatmap(mean_img, cfg, vmin, vmax)

    if not quiet:
        print(f"[OK] Saved bar chart to: {bar_path}")
        print(f"[OK] Saved ROI heatmap to: {heatmap_path}")

    return PlotOutputs(roi_df=roi_df, bar_path=bar_path, heatmap_path=heatmap_path)


if __name__ == "__main__":
    from nilearn import datasets

    # ===== CONFIGURATION: Set the number of top ROIs to display =====
    TOP_N_ROIS = 27  # Change this value to show different number of ROIs
    TOP_N_ROIS = 100
    
    # Load default reference grid (MNI152, 2 mm)
    ref_img = datasets.load_mni152_template(resolution=2)

    # Call the revised modular function
    outputs = plot_shap_from_csv_modular(
        shap_csv_path="/home/junfu.cheng/SMILE/github/j_map_fake/J_MAP_RESULTS/results_2025_09_11_12_25_06/seed_154/shap_table.csv",
        reference_img=ref_img,
        metric_name="SHAP",
        background="white",
        top_n_regions=TOP_N_ROIS,  # Use the constant here
        output_dir="figure_brain",
        atlas_name="cort-maxprob-thr25-2mm",
        subcortical_atlas_name="sub-maxprob-thr25-2mm",
        include_subcortical=True,
        display_mode="ortho",
        cut_coords=None,
        quiet=False,
    )

    print(f"[OK] Bar chart saved to: {outputs.bar_path}")
    print(f"[OK] ROI heatmap saved to: {outputs.heatmap_path}")
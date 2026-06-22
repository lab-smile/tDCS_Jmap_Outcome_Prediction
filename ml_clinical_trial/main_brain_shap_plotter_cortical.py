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
    background: str = "black"   # "black" or "white"
    top_n_regions: Optional[int] = None
    output_dir: str = "figure_brain"
    heatmap_filename: str = "roi_shap_heatmap.png"
    bar_filename_prefix: str = "roi_contrib"  # full name built from metric
    atlas_name: str = "cort-maxprob-thr25-2mm"  # Harvard–Oxford (cortical)
    # display_mode: "ortho", "x", "y", "z", etc.
    display_mode: str = "ortho"
    # Optional fixed cut coordinates (x,y,z). If None, nilearn auto-selects.
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

    rows: List[Tuple[int, str, float]] = []
    for roi_idx in range(1, len(atlas_labels)):
        roi_name = str(atlas_labels[roi_idx])
        if roi_name.lower() in {"background", "unknown", "unlabeled"}:
            continue

        mask = (atlas_arr == roi_idx)
        if not np.any(mask):
            continue

        vals = abs_arr[mask]
        if np.all(np.isnan(vals)):
            continue

        rows.append((roi_idx, roi_name, float(np.nanmean(vals))))

    if not rows:
        raise ValueError("No valid ROIs found with |SHAP| values.")

    df = pd.DataFrame(rows, columns=["roi_index", "roi_name", "mean_abs_shap"])
    df.sort_values("mean_abs_shap", ascending=False, inplace=True, ignore_index=True)
    return df


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
    ax.set_xlabel("Mean |SHAP| per ROI", color=text_color)
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

    # 3) Atlas (aligned to grid)
    atlas_img, atlas_arr, atlas_labels = fetch_resampled_harvard_oxford(ref_img, cfg.atlas_name)

    # 4) Mean(|SHAP|) per ROI
    roi_df = compute_mean_abs_by_roi(shap_abs_img, atlas_arr, atlas_labels)

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
    TOP_N_ROIS = 10  # Change this value to show different number of ROIs
    
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
        display_mode="ortho",
        cut_coords=None,   # e.g., (0, 0, 22) if you want fixed cuts
        quiet=False,
    )

    print(f"[OK] Bar chart saved to: {outputs.bar_path}")
    print(f"[OK] ROI heatmap saved to: {outputs.heatmap_path}")
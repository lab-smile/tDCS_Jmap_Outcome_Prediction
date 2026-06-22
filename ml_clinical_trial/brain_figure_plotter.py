import os
import ast
from nilearn import plotting, datasets, image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from nilearn.image import new_img_like
from matplotlib import colors, colormaps  # <-- use colormaps instead of cm.get_cmap
import warnings

from nilearn.image import new_img_like

def plot_harvard_oxford(output_path="brain_parcellation.png", quality="high", quiet=True):
    # Optional: silence known future warnings cleanly
    if quiet:
        warnings.filterwarnings(
            "ignore",
            category=FutureWarning,
            message=".*force_resample.*will be set to 'True'.*"
        )
        warnings.filterwarnings(
            "ignore",
            category=FutureWarning,
            message=".*copy the header of the input image.*"
        )

    atlas = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
    atlas_img = atlas.maps
    bg_1mm = datasets.load_mni152_template(resolution=1)

    # Explicit future defaults:
    if quality in ["high", "smooth"]:
        atlas_img = image.resample_to_img(
            atlas_img,
            bg_1mm,
            interpolation="nearest",
            force_resample=True,   # <-- future default, set explicitly now
            copy_header=True       # <-- future default, set explicitly now
        )
        bg_img = bg_1mm
    else:
        bg_img = None

    fig = plt.figure(figsize=(10, 10))
    interp = "nearest" if quality in ["fast", "high"] else "linear"

    display = plotting.plot_roi(
        atlas_img,
        bg_img=bg_img,
        title=f"Harvard–Oxford Atlas ({quality})",
        draw_cross=False,
        cmap="tab20",
        colorbar=True,
        resampling_interpolation=interp,
        figure=fig,
    )

    dpi = 200 if quality == "fast" else 600
    display.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {quality}-quality figure to: {output_path}")

def plot_roi_means(
    data,
    metric_name="Mean",
    output_dir="figure_brain",
    top_n=None,
    reference_img=None,
    quality="high",
    background="white",   # <-- "white" or "black"
    quiet=True
):
    """
    Compute mean per Harvard–Oxford ROI and save:
      - roi_<metric>_heatmap.png  (brain heatmap; background white/black)
      - roi_<metric>_bar.png      (horizontal bar chart; highest at top)

    Parameters
    ----------
    data : nibabel.Nifti1Image or np.ndarray
        3D brain volume. If np.ndarray, provide `reference_img`.
    metric_name : str
        Label for the mean statistic (e.g., "T-value", "Random").
    output_dir : str
        Folder to save figures.
    top_n : int or None
        Show only the top N ROIs in the bar chart if set.
    reference_img : nibabel.Nifti1Image or None
        Required when `data` is a NumPy array.
    quality : {"fast", "high", "smooth"}
        Controls anatomical background resolution (1mm for high/smooth, 2mm for fast).
    background : {"white", "black"}
        Figure background for both heatmap and bar chart.
    quiet : bool
        Suppress known future warnings from nilearn.
    """
    if quiet:
        warnings.filterwarnings("ignore", category=FutureWarning, message=".*force_resample.*")
        warnings.filterwarnings("ignore", category=FutureWarning, message=".*copy the header.*")

    os.makedirs(output_dir, exist_ok=True)

    # --- Normalize input to NIfTI ---
    if hasattr(data, "shape") and hasattr(data, "affine"):  # NIfTI-like
        data_img = data
        data_arr = image.get_data(data_img)
    else:
        if reference_img is None:
            raise ValueError("If `data` is a NumPy array, provide `reference_img` (NIfTI).")
        if data.shape != reference_img.shape:
            raise ValueError("`data` shape must match `reference_img` shape.")
        data_img = new_img_like(reference_img, data)
        data_arr = data

    # --- Atlas and labels ---
    atlas = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
    atlas_img_native = atlas.maps
    atlas_labels = list(atlas.labels)  # index 0 = "Background"

    # --- Resample atlas to data grid (nearest preserves labels) ---
    atlas_img = image.resample_to_img(
        atlas_img_native, data_img, interpolation="nearest",
        force_resample=True, copy_header=True
    )
    atlas_arr = image.get_data(atlas_img)

    # --- Compute per-ROI mean ---
    roi_means = []
    for roi_idx in range(1, len(atlas_labels)):
        mask = (atlas_arr == roi_idx)
        if not np.any(mask):
            continue
        vals = data_arr[mask]
        m = np.nan if np.all(np.isnan(vals)) else np.nanmean(vals)
        if not np.isnan(m):
            roi_means.append((roi_idx, atlas_labels[roi_idx], m))

    # Sort descending by mean (highest first)
    roi_means.sort(key=lambda x: x[2], reverse=True)

    # --- Build mean map for heatmap (NaN outside ROIs so only brain shows) ---
    mean_map = np.full(atlas_arr.shape, np.nan, dtype=float)  # NaN -> transparent in plot_img
    for roi_idx, _, m in roi_means:
        mean_map[atlas_arr == roi_idx] = m
    mean_img = new_img_like(atlas_img, mean_map)

    # Background anatomy template
    bg_img = datasets.load_mni152_template(resolution=1 if quality in ["high", "smooth"] else 2)

    # --- Build mean map for heatmap with transparent outside-ROI ---
    # Compute vmin/vmax first from roi_means
    vmin = min(m for _, _, m in roi_means)
    vmax = max(m for _, _, m in roi_means)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax == vmin:
        # guard against degenerate cases
        vmin, vmax = float(np.nanmin([m for _, _, m in roi_means])), float(np.nanmax([m for _, _, m in roi_means]))
        if vmax == vmin:
            vmax = vmin + 1e-6

    # Use a sentinel below vmin; color “under” = fully transparent
    sentinel = vmin - (vmax - vmin + np.finfo(float).eps)
    mean_map = np.full(atlas_arr.shape, sentinel, dtype=float)
    for roi_idx, _, m in roi_means:
        mean_map[atlas_arr == roi_idx] = m
    mean_img = new_img_like(atlas_img, mean_map)

    # Anatomical background
    bg_img = datasets.load_mni152_template(resolution=1 if quality in ["high", "smooth"] else 2)

    # Colormap with transparent "under" (outside-ROI sentinel)
    from matplotlib import colormaps
    cmap = colormaps.get_cmap("coolwarm").with_extremes(under=(0, 0, 0, 0))

    # Background color choice
    bg_color = "black" if background.lower() == "black" else "white"
    black_bg = (bg_color == "black")

    # --- Brain heatmap ---
    brain_fig = plt.figure(figsize=(10, 10), facecolor=bg_color)
    brain_disp = plotting.plot_img(
        mean_img,
        bg_img=bg_img,
        cmap=cmap,
        vmin=vmin, vmax=vmax,
        colorbar=True,
        figure=brain_fig,
        black_bg=black_bg,
        title=f"{metric_name} per ROI (Harvard–Oxford)"
    )

    brain_path = os.path.join(output_dir, f"roi_{metric_name.replace(' ', '_').lower()}_heatmap.png")
    # IMPORTANT: do NOT pass facecolor here; we already set figure facecolor above.
    brain_disp.savefig(brain_path, dpi=600, bbox_inches="tight")
    plt.close(brain_fig)

def plot_shap_from_csv(
    shap_csv_path,
    reference_img,
    output_dir="figure_brain",
    metric_name="SHAP",
    background="black",      # "black" or "white"
    quality="high",          # kept for API; heatmap uses MNI-1mm
    top_n_regions=None,      # e.g., 25 for the bar chart
    quiet=True
):
    """
    Reads SHAP coordinate CSV, builds an |SHAP| volume, computes Harvard–Oxford
    ROI contributions = mean(|SHAP|) per ROI, and outputs:

      - roi_contrib_<metric>_bar.png  (bar colors match heatmap colors)
      - roi_shap_heatmap.png          (tight layout)

    Background / unlabeled voxels are excluded from the bar chart.
    """
    # ---- Load reference grid ----
    if isinstance(reference_img, str):
        ref_img = nib.load(reference_img)
    else:
        ref_img = reference_img
    ref_shape = ref_img.shape
    os.makedirs(output_dir, exist_ok=True)

    # ---- Load CSV ----
    df = pd.read_csv(shap_csv_path)
    req_cols = {"voxel_ijk", "mean_abs_shap_voxel"}
    missing = [c for c in req_cols if c not in df.columns]
    if missing:
        raise ValueError(f"shap_table.csv is missing required column(s): {missing}")

    # Parse voxel_ijk safely
    def _parse_ijk(x):
        if isinstance(x, (tuple, list, np.ndarray)) and len(x) >= 3:
            return int(x[0]), int(x[1]), int(x[2])
        if isinstance(x, str):
            try:
                tup = ast.literal_eval(x)
                return int(tup[0]), int(tup[1]), int(tup[2])
            except Exception:
                s = x.strip().strip("()[]")
                parts = s.split(",")
                if len(parts) >= 3:
                    return int(parts[0]), int(parts[1]), int(parts[2])
        raise ValueError(f"Cannot parse voxel_ijk entry: {x!r}")

    ijk_list = df["voxel_ijk"].apply(_parse_ijk).tolist()
    shap_abs_vals = df["mean_abs_shap_voxel"].to_numpy(dtype=float)

    # ---- Build sparse |SHAP| image on reference grid ----
    shap_abs_arr = np.full(ref_shape, np.nan, dtype=float)
    for (i, j, k), v in zip(ijk_list, shap_abs_vals):
        if 0 <= i < ref_shape[0] and 0 <= j < ref_shape[1] and 0 <= k < ref_shape[2]:
            shap_abs_arr[i, j, k] = v
    shap_abs_img = new_img_like(ref_img, shap_abs_arr)

    # ---- Harvard–Oxford atlas on the same grid ----
    atlas = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
    atlas_img = image.resample_to_img(
        atlas.maps, ref_img, interpolation="nearest",
        force_resample=True, copy_header=True
    )
    atlas_arr = image.get_data(atlas_img)
    atlas_labels = list(atlas.labels)  # 0 = Background

    # ---- Mean(|SHAP|) per ROI, skipping background/unlabeled ----
    roi_rows = []
    for roi_idx in range(1, len(atlas_labels)):
        roi_name = str(atlas_labels[roi_idx])
        if roi_name.lower() in {"background", "unknown", "unlabeled"}:
            continue
        mask = (atlas_arr == roi_idx)
        if not np.any(mask):
            continue
        vals = shap_abs_arr[mask]
        if np.all(np.isnan(vals)):
            continue
        m = float(np.nanmean(vals))
        roi_rows.append((roi_idx, roi_name, m))

    if not roi_rows:
        raise ValueError("No valid ROIs found with |SHAP| values.")

    roi_rows.sort(key=lambda x: x[2], reverse=True)
    roi_df = pd.DataFrame(roi_rows, columns=["roi_index", "roi_name", "mean_abs_shap"])

    # ---- Shared color mapping (cmap + vmin/vmax) ----
    cmap = colormaps.get_cmap("coolwarm")
    vmin = float(roi_df["mean_abs_shap"].min())
    vmax = float(roi_df["mean_abs_shap"].max())
    # Guard against degenerate range
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax == vmin:
        vmax = vmin + 1e-6
    norm = colors.Normalize(vmin=vmin, vmax=vmax)

    # ---- Bar chart (colors correspond to heatmap) ----
    if top_n_regions is not None:
        roi_df_plot = roi_df.head(int(top_n_regions)).copy()
    else:
        roi_df_plot = roi_df.copy()

    bg_color = "black" if str(background).lower() == "black" else "white"
    text_color = "white" if bg_color == "black" else "black"

    fig, ax = plt.subplots(
        figsize=(10, max(4, 0.35 * len(roi_df_plot))),
        facecolor=bg_color
    )

    bar_colors = [cmap(norm(v)) for v in roi_df_plot["mean_abs_shap"][::-1]]
    ax.barh(
        roi_df_plot["roi_name"][::-1],
        roi_df_plot["mean_abs_shap"][::-1],
        color=bar_colors,
        edgecolor=text_color,
        linewidth=0.5
    )
    ax.set_xlabel("Mean |SHAP| per ROI", color=text_color)
    ax.set_title(f"Mean ROI Contributions ({metric_name}) — Harvard–Oxford", color=text_color)
    ax.tick_params(colors=text_color)
    for spine in ax.spines.values():
        spine.set_color(text_color)
    fig.tight_layout()

    slug = metric_name.replace(" ", "_").lower()
    bar_path = os.path.join(output_dir, f"roi_contrib_{slug}_bar.png")
    fig.savefig(bar_path, dpi=600, bbox_inches="tight")
    plt.close(fig)

    # ---- ROI heatmap using the SAME cmap/vmin/vmax, tight layout ----
    # Build per-voxel map that is sentinel outside atlas so it's transparent
    sentinel = vmin - (vmax - vmin + np.finfo(float).eps)
    mean_map = np.full(atlas_arr.shape, sentinel, dtype=float)

    # Fill each ROI with its mean(|SHAP|) so colors match the bar chart
    roi_mean_by_idx = dict(zip(roi_df["roi_index"], roi_df["mean_abs_shap"]))
    for roi_idx, m in roi_mean_by_idx.items():
        mean_map[atlas_arr == roi_idx] = m

    mean_img = new_img_like(atlas_img, mean_map)

    # Transparent "under" so non-ROI voxels don't show; keep cmap otherwise identical
    cmap_for_img = cmap.with_extremes(under=(0, 0, 0, 0))

    # Higher-res anatomical background for looks
    bg_img = datasets.load_mni152_template(resolution=1)

    # --- ROI heatmap with larger brain panel ---
    roi_fig = plt.figure(figsize=(12, 7), facecolor=bg_color)
    black_bg = (bg_color == "black")

    # 1) A large axis for the brain panel (fills most of the figure)
    #    [left, bottom, width, height] in figure coords
    brain_ax = roi_fig.add_axes([0.04, 0.06, 0.80, 0.88])

    disp = plotting.plot_img(
        mean_img,
        bg_img=bg_img,
        cmap=cmap_for_img,
        vmin=vmin,
        vmax=vmax,
        colorbar=True,          # keep colorbar so we can then reposition it
        figure=roi_fig,
        axes=brain_ax,          # <— tell nilearn to draw into our big axis
        black_bg=black_bg,
        title="Mean |SHAP| per ROI (Harvard–Oxford)",
        display_mode="ortho",   # same 3 views
        # cut_coords=(0, 0, 22),  # or leave None for auto
    )

    # 2) Make the colorbar small and out of the way so the brain stays large
    #    The colorbar is the last axis added to the figure
    cbar_ax = roi_fig.axes[-1]
    cbar_ax.set_position([0.87, 0.25, 0.025, 0.5])  # thin, centered colorbar

    # 3) Save tightly (no extra padding)
    roi_path = os.path.join(output_dir, "roi_shap_heatmap.png")
    disp.savefig(roi_path, dpi=600, bbox_inches="tight", pad_inches=0)
    plt.close(roi_fig)

    print(f"[OK] Saved bar chart to: {bar_path}")
    print(f"[OK] Saved ROI heatmap to: {roi_path}")

def plot_all_shap_from_experiment(
    experiment_dir,
    reference_img,
    output_dir="figure_brain",
    metric_name="SHAP",
    background="black",
    quality="high",
    top_n_regions=None,
    quiet=True,
    csv_name="shap_table.csv"
):
    """
    For each subfolder under experiment_dir that starts with 'seed',
    call plot_shap_from_csv and save its figures into a matching subfolder
    inside output_dir (e.g., output_dir/seed123).

    Args:
        experiment_dir (str): Path containing seed folders.
        reference_img (str or nibabel image): Reference brain image.
        output_dir (str): Base folder for figures.
        metric_name, background, quality, top_n_regions, quiet: passed through.
        csv_name (str): Default shap table filename.
    """
    if not os.path.isdir(experiment_dir):
        raise NotADirectoryError(f"Not a directory: {experiment_dir}")

    seed_dirs = [
        d for d in sorted(os.listdir(experiment_dir))
        if d.startswith("seed") and os.path.isdir(os.path.join(experiment_dir, d))
    ]

    if not seed_dirs and not quiet:
        print(f"[WARN] No 'seed*' subfolders found in {experiment_dir}")

    results = {"success": [], "fail": []}

    for seed in seed_dirs:
        shap_csv_path = os.path.join(experiment_dir, seed, csv_name)
        if not os.path.isfile(shap_csv_path):
            if not quiet:
                print(f"[SKIP] {seed}: no {csv_name}")
            continue

        out_dir_seed = os.path.join(output_dir, seed)
        os.makedirs(out_dir_seed, exist_ok=True)

        try:
            plot_shap_from_csv(
                shap_csv_path=shap_csv_path,
                reference_img=reference_img,
                output_dir=out_dir_seed,
                metric_name=metric_name,
                background=background,
                quality=quality,
                top_n_regions=top_n_regions,
                quiet=quiet,
            )
            results["success"].append(seed)
        except Exception as e:
            results["fail"].append((seed, str(e)))
            if not quiet:
                print(f"[ERROR] {seed}: {e}")

    if not quiet:
        print(f"[DONE] {len(results['success'])} ok, {len(results['fail'])} failed")
    return results



def plot_aggregate_shap_from_experiment(
    experiment_dir,
    reference_img,
    output_dir="figure_brain",
    background="black",        # "black" or "white"
    top_n_regions=None,        # e.g., 25 for the bar chart (stats still computed for all)
    quiet=True,
    csv_name="shap_table.csv"
):
    """
    Aggregate ROI contributions across all 'seed*' subfolders inside `experiment_dir`.
    Each seed folder must contain a per-voxel CSV (default: 'shap_table.csv') with
    columns ['voxel_ijk', 'mean_abs_shap_voxel'].

    Outputs (in `output_dir/aggregate/`):
      - roi_contrib_shap_bar.png     (bar: mean ± SD across seeds, error bars)
      - roi_shap_heatmap.png         (heatmap of mean across seeds)
      - roi_shap_heatmap_top6.png    (heatmap showing only top-6 ROIs)
      - stats_table.csv              (Mean, SD, N, 95% CI; also formatted columns)
    """
    # --- Local imports to keep the function self-contained ---
    import os, ast
    import numpy as np
    import pandas as pd
    import nibabel as nib
    import matplotlib.pyplot as plt
    from matplotlib import colors, colormaps
    from nilearn import plotting, image, datasets

    # ---- Prepare reference image ----
    if isinstance(reference_img, str):
        ref_img = nib.load(reference_img)
    else:
        ref_img = reference_img
    ref_shape = ref_img.shape

    # ---- Gather seed directories ----
    if not os.path.isdir(experiment_dir):
        raise NotADirectoryError(f"Not a directory: {experiment_dir!r}")

    seed_dirs = [
        d for d in sorted(os.listdir(experiment_dir))
        if d.startswith("seed") and os.path.isdir(os.path.join(experiment_dir, d))
    ]
    if not seed_dirs:
        raise FileNotFoundError(f"No 'seed*' subfolders found in {experiment_dir}")

    # ---- Robust voxel_ijk parser ----
    def _parse_ijk(x):
        if isinstance(x, (tuple, list, np.ndarray)) and len(x) >= 3:
            return int(x[0]), int(x[1]), int(x[2])
        if isinstance(x, str):
            try:
                tup = ast.literal_eval(x)
                return int(tup[0]), int(tup[1]), int(tup[2])
            except Exception:
                s = x.strip().strip("()[]")
                parts = s.split(",")
                if len(parts) >= 3:
                    return int(parts[0]), int(parts[1]), int(parts[2])
        raise ValueError(f"Cannot parse voxel_ijk entry: {x!r}")

    # ---- Load/prepare atlas on same grid ----
    atlas = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
    atlas_img = image.resample_to_img(
        atlas.maps, ref_img, interpolation="nearest",
        force_resample=True, copy_header=True
    )
    atlas_arr = image.get_data(atlas_img)
    atlas_labels = list(atlas.labels)  # 0 = Background

    # ---- For each seed: build |SHAP| volume and compute mean(|SHAP|) per ROI ----
    per_seed_roi = []  # columns: roi_index, roi_name, mean_abs_shap, seed

    for seed in seed_dirs:
        shap_csv_path = os.path.join(experiment_dir, seed, csv_name)
        if not os.path.isfile(shap_csv_path):
            if not quiet:
                print(f"[SKIP] {seed}: missing {csv_name}")
            continue

        df = pd.read_csv(shap_csv_path)
        req_cols = {"voxel_ijk", "mean_abs_shap_voxel"}
        missing = [c for c in req_cols if c not in df.columns]
        if missing:
            if not quiet:
                print(f"[SKIP] {seed}: {csv_name} missing columns {missing}")
            continue

        ijk_list = df["voxel_ijk"].apply(_parse_ijk).tolist()
        shap_abs_vals = pd.to_numeric(df["mean_abs_shap_voxel"], errors="coerce").to_numpy(dtype=float)

        shap_abs_arr = np.full(ref_shape, np.nan, dtype=float)
        for (i, j, k), v in zip(ijk_list, shap_abs_vals):
            if 0 <= i < ref_shape[0] and 0 <= j < ref_shape[1] and 0 <= k < ref_shape[2]:
                shap_abs_arr[i, j, k] = v

        # Mean(|SHAP|) per ROI (skip background/unlabeled)
        rows = []
        for roi_idx in range(1, len(atlas_labels)):
            roi_name = str(atlas_labels[roi_idx])
            if roi_name.lower() in {"background", "unknown", "unlabeled"}:
                continue
            mask = (atlas_arr == roi_idx)
            if not np.any(mask):
                continue
            vals = shap_abs_arr[mask]
            if np.all(np.isnan(vals)):
                continue
            m = float(np.nanmean(vals))
            rows.append((roi_idx, roi_name, m))

        if not rows:
            if not quiet:
                print(f"[WARN] {seed}: no valid ROIs with |SHAP| values")
            continue

        df_seed = pd.DataFrame(rows, columns=["roi_index", "roi_name", "mean_abs_shap"])
        df_seed["seed"] = seed
        per_seed_roi.append(df_seed)

        if not quiet:
            print(f"[OK] {seed}: {len(df_seed)} ROIs summarized")

    if not per_seed_roi:
        raise RuntimeError("No valid per-seed ROI summaries were computed.")

    # ---- Aggregate across seeds: mean, SD, N, 95% CI ----
    all_roi = pd.concat(per_seed_roi, ignore_index=True)
    grouped = all_roi.groupby(["roi_index", "roi_name"])["mean_abs_shap"]

    agg = grouped.agg(["mean", "std", "count"]).reset_index()
    agg.rename(columns={"count": "n"}, inplace=True)

    # 95% CI (t-based if possible; fallback to z=1.96)
    try:
        from scipy.stats import t
        crit = t.ppf(0.975, np.maximum(agg["n"] - 1, 1))
    except Exception:
        crit = np.full(len(agg), 1.96)

    se = agg["std"] / np.sqrt(agg["n"]).replace(0, np.nan)
    ci_lower = agg["mean"] - crit * se
    ci_upper = agg["mean"] + crit * se

    agg["ci_lower"] = ci_lower
    agg["ci_upper"] = ci_upper

    # Nicely formatted columns
    def _fmt(x):
        return f"{x:.4g}" if np.isfinite(x) else "NA"

    agg["Mean \u00B1 SD"] = [
        f"{_fmt(m)} \u00B1 {_fmt(s)}" for m, s in zip(agg["mean"], agg["std"])
    ]
    agg["95% CI (Lower\u2013Upper)"] = [
        f"{_fmt(l)}\u2013{_fmt(u)}" for l, u in zip(agg["ci_lower"], agg["ci_upper"])
    ]

    # ---- Output directory ----
    agg_dir = os.path.join(output_dir, "aggregate")
    os.makedirs(agg_dir, exist_ok=True)

    # ---- Color mapping (ROBUST + NaN transparency) ----
    # Use percentiles so the scale isn't squashed into pale blue.




if __name__ == "__main__":
    import os
    import numpy as np
    from nilearn import datasets

    # 0) Output directory
    output_dir = "figure_brain"
    os.makedirs(output_dir, exist_ok=True)

    # # 1) Plot atlas variants
    # plot_harvard_oxford(os.path.join(output_dir, "atlas_fast.png"),   quality="fast")
    # plot_harvard_oxford(os.path.join(output_dir, "atlas_high.png"),   quality="high")
    # plot_harvard_oxford(os.path.join(output_dir, "atlas_smooth.png"), quality="smooth")

    # # 2) Demo data for ROI means (NumPy array + reference grid)
    # np.random.seed(0)  # reproducible example
    # ref_img = datasets.load_mni152_template(resolution=2)
    # arr = np.random.randn(*ref_img.shape)

    # # 3) Compute ROI means + visuals
    # #    Highest bars at top; black background brain & chart; top 25 ROIs in chart
    # plot_roi_means(
    #     arr,
    #     metric_name="T-value",
    #     reference_img=ref_img,
    #     output_dir=output_dir,
    #     top_n=25,
    #     background="black",   # or "white"
    #     quality="high",       # heatmap background resolution
    #     quiet=True            # suppress future warnings
    # )

    # Assuming you already produced shap_table.csv with get_coordinate_table_mni_shap(...)
    from nilearn import datasets
    ref_img = datasets.load_mni152_template(resolution=2)

    get_csv = True
    if get_csv == True:
        plot_shap_from_csv(
            shap_csv_path="/home/junfu.cheng/SMILE/github/j_map_fake/J_MAP_RESULTS/results_2025_09_11_12_25_06/seed_154/shap_table.csv",
            reference_img=ref_img,
            output_dir="figure_brain",
            metric_name="SHAP",
            #use_abs=True,
            background="white",
            quality="high",
            top_n_regions=10
        )

    show_j_map = False
    if show_j_map == True:
        from .model_dataset_preparator import ModelDatasetPreparator
        relative_path = '../../../data_generation_log/act_data'
        experiment_dir = '/home/junfu.cheng/SMILE/github/j_map_2025_8_16/figure_brain'
        file_name='act_data_generated.csv'
        dict_filename='act_data_dict_generated.csv'
        target_feature='stai_state_score'
        responder_criteria='above_median_decrease_in_severe'
        group_var_name='Group_tp0'
        group_value=[4],
        visit_times=['0', '1']

        model_dataset_preparator = ModelDatasetPreparator(
                    relative_path,
                    experiment_dir,
                    file_name=file_name,
                    dict_filename=dict_filename,
                    target_feature=target_feature,
                    responder_criteria=responder_criteria,
                    group_var_name=group_var_name,
                    group_value=group_value,
                    visit_times=visit_times,
                    printer=print,
                )
        (X_train, X_test, y_train, y_test,
            features, numerical_features, categorical_features, jmap_features) = \
                model_dataset_preparator.prepare_train_test_data_in_severe_state_anxiety_in_jmap()


        print(X_train.head())
        print('print(X_train.dtypes)')
        print(X_train['jmap_tp1'].apply(type).head())

        from nilearn import datasets
        from .brainviz import BrainVisualizer  # <- your new module

        # Load reference template
        ref_img = datasets.load_mni152_template(resolution=2)

        # Parameters
        out_dir = "/home/junfu.cheng/SMILE/github/j_map_fake/figure_brain/figure/jmap_visulize"
        roi_name = "Temporal Fusiform Cortex, posterior division"

        X_all = pd.concat([X_train, X_test], axis=0)  # keep existing indexes
        viz = BrainVisualizer()
        value_range = viz.compute_value_range(X_all, roi_name, ref_img )
        print(f"Value range for {roi_name}: {value_range}")
        if value_range[0] < 0:
            print("negative current density magnitude detected, set the min value to 0")
            value_range = (0, value_range[1])
        # participant_id = 105256

        def get_jmap_visualied(X, ref_img, roi_name, value_range, out_dir, set_name, y):
            for participant_id in X.index.tolist():
                label = 'responder' if y.loc[participant_id] == 1 else 'non_responder'
                out_dir_participant = os.path.join(out_dir, set_name+label, str(participant_id))
                # Instantiate the visualizer
                viz = BrainVisualizer()

                # 1. Retrieve the jmap array
                arr = viz.get_jmap_array(participant_id, X)

                # 2. Visualize and save figures (PNG/SVG/PDF)
                paths = viz.visualize_brain_image(
                    img=arr,
                    out_dir = out_dir_participant,
                    roi_name=roi_name,            # or roi_name if you want ROI masking
                    reference_img=ref_img,
                    save=True,
                    participant_id = participant_id
                )
                saved3d_path = viz.visualize_brain_3d_grid(
                    arr,
                    out_dir = out_dir_participant,
                    participant_id = participant_id,
                    roi_name=roi_name,            # or roi_name if you want ROI masking
                    reference_img=ref_img,
                    value_range = value_range
                )
                histogram_path = viz.visualize_brain_3d_histogram(
                    arr,
                    out_dir = out_dir_participant,
                    participant_id = participant_id,
                    roi_name=roi_name,            # or roi_name if you want ROI masking
                    reference_img=ref_img,
                    value_range = value_range
                )
                cloud_path = viz.visualize_brain_3d_cloud(
                    arr,
                    out_dir = out_dir_participant,
                    participant_id = participant_id,
                    roi_name=roi_name,            # or roi_name if you want ROI masking
                    reference_img=ref_img,
                    value_range = value_range
                )

                print("Saved file paths:")
                print(paths)
                print("Saved 3D file paths:")
                print(saved3d_path)
                print("Saved histogram paths:")
                print(histogram_path)
                print("Svaed cloud path:")
                print(cloud_path)

        get_jmap_visualied(X_train, ref_img, roi_name, value_range, out_dir, 'training', y_train)
        get_jmap_visualied(X_test, ref_img, roi_name, value_range, out_dir, 'testing', y_test)
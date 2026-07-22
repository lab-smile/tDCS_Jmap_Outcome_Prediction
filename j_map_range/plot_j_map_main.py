#!/usr/bin/env python3
"""
Current-density visualisation for brain tDCS simulations.

Generates axial / sagittal / coronal current-density maps (magnitude +
in-plane direction arrows) for a list of subjects, using a shared colour
scale across every figure.

For every slice, TWO figure groups are written:
  1. the standard figures        ->  <name>.png / .pdf
  2. a "font bigger" variant      ->  <name>_font_bigger.png / .pdf
     Larger text, optionally all-bold, and with the
     "[shared colorbar: ... A/m²]" line removed from the title.

--------------------------------------------------------------------------
Debugging aids
--------------------------------------------------------------------------
* All console output goes through the `logging` module. Set LOG_LEVEL to
  logging.DEBUG for verbose per-step output, or logging.WARNING to quieten it.
* Numeric work (`compute_slice_render_data`) is separated from drawing
  (`draw_figure`), so a styling bug can be inspected/fixed without re-running
  the pixel pipeline, and the standard + font-bigger figures share one
  computation.
* Subjects with missing files are detected up front and skipped with a warning.
* Each subject / plane is processed inside a try/except: one failure is logged
  (with traceback) and skipped instead of aborting the whole run. A summary of
  failures is printed at the end.
"""

import os
import logging
import traceback
from dataclasses import dataclass

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib import cm
from scipy.ndimage import gaussian_filter, binary_closing


# ============================================================
# Logging (single debug switch)
# ============================================================
LOG_LEVEL = logging.INFO       # logging.DEBUG = verbose, logging.WARNING = quiet
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("current_density")


# ============================================================
# User settings
# ============================================================
subject_ids = [
    "107802",
    "105601",
    "110081",
    "116036",
]
base_dir = "/blue/camctrp/working/junfu.cheng/roast_output_spm_registration_auto_accelerated"
output_dir = "/blue/camctrp/working/junfu.cheng/roast_output_spm_registration_auto_accelerated/current_density_axial_plots"

# NIfTI file names inside each subject directory
J_FILENAME     = "wT1_tDCSLAB_Jbrain.nii"
PHI_FILENAME   = "wT1_tDCSLAB_PhiBrain.nii"
THETA_FILENAME = "wT1_tDCSLAB_ThetaBrain.nii"

slice_index = {
    'axial': 40,
    'sagittal': 60,
    'coronal': 48,
}

plot_planes = ['axial', 'sagittal', 'coronal']

# Plot settings
magnitude_threshold = 0.01
arrow_step = 6
arrow_scale = 0.30
arrow_width = 0.001
arrow_color = 'black'
arrow_alpha = 0.2
arrow_edge_color = 'black'
arrow_edge_width = 0.8

# Smoothing settings
apply_smoothing = True
gaussian_sigma = 1.5
edge_smoothing_sigma = 0.9
interpolation = 'bilinear'

# Morphological operations
apply_morphology = True
morph_iterations = 2

# Color settings
vmin_percentile = 5
vmax_percentile = 99.5

# Angles setting
angles_are_degrees = False


# ============================================================
# STANDARD figure-group fonts (kept identical to the original script)
# ------------------------------------------------------------
# In the original, the title and colour-bar label were bold and the tick
# numbers were normal weight; those defaults are preserved here.
# ============================================================
STD_TITLE_FONT_SIZE      = 15
STD_CBAR_LABEL_FONT_SIZE = 14
STD_CBAR_TICK_FONT_SIZE  = 11
STD_SHOW_COLORBAR_RANGE  = True     # keep the "[shared colorbar: ...]" title line


# ============================================================
# "FONT BIGGER" figure-group settings
# ------------------------------------------------------------
# A second copy of every figure is written with a "_font_bigger" suffix using
# the settings below.
# ============================================================
GENERATE_FONT_BIGGER_GROUP      = True    # turn the extra group on/off
FONT_BIGGER_FONT_SIZE           = 30      # ONE size for every letter & number
FONT_BIGGER_ALL_BOLD            = True    # make every letter & number bold (default: True)
FONT_BIGGER_SHOW_COLORBAR_RANGE = False   # omit "[shared colorbar: ...]" line in title


# ============================================================
# Figure-style definition (what differs between the two groups)
# ============================================================
@dataclass
class FigureStyle:
    filename_suffix: str          # "" or "_font_bigger"
    title_font_size: float
    cbar_label_font_size: float
    cbar_tick_font_size: float
    title_weight: str             # 'bold' or 'normal'
    label_weight: str             # 'bold' or 'normal'
    tick_bold: bool               # bold the colour-bar tick NUMBERS?
    show_colorbar_range: bool     # show the "[shared colorbar: ...]" title line?
    simple_title: bool            # short title, e.g. "Axial (z = 40)" only?


def build_styles():
    """Return the list of FigureStyle objects to render for every slice."""
    styles = [
        FigureStyle(
            filename_suffix="",
            title_font_size=STD_TITLE_FONT_SIZE,
            cbar_label_font_size=STD_CBAR_LABEL_FONT_SIZE,
            cbar_tick_font_size=STD_CBAR_TICK_FONT_SIZE,
            title_weight="bold",           # original behaviour
            label_weight="bold",           # original behaviour
            tick_bold=False,               # original behaviour
            show_colorbar_range=STD_SHOW_COLORBAR_RANGE,
            simple_title=False,            # keep the full multi-line title
        )
    ]
    if GENERATE_FONT_BIGGER_GROUP:
        w = "bold" if FONT_BIGGER_ALL_BOLD else "normal"
        styles.append(
            FigureStyle(
                filename_suffix="_font_bigger",
                title_font_size=FONT_BIGGER_FONT_SIZE,
                cbar_label_font_size=FONT_BIGGER_FONT_SIZE,
                cbar_tick_font_size=FONT_BIGGER_FONT_SIZE,
                title_weight=w,
                label_weight=w,
                tick_bold=FONT_BIGGER_ALL_BOLD,
                show_colorbar_range=FONT_BIGGER_SHOW_COLORBAR_RANGE,
                simple_title=True,         # title is just "Axial (z = 40)" etc.
            )
        )
    return styles


# ------------------------------------------------------------
# Custom colormap defined via RGB control points
# (equivalent to "coolwarm" but explicit, so it can be ported to
#  ParaView / MATLAB / ITK-SNAP, etc.)
# Control points: (position_0_to_1, R, G, B) — all values in [0, 1]
# ------------------------------------------------------------
CUSTOM_CMAP_RGB_NODES = [
    # pos    R       G       B
    (0.000, 0.017,  0.176,  0.612),   # deep blue
    (0.125, 0.216,  0.431,  0.776),
    (0.250, 0.435,  0.663,  0.890),
    (0.375, 0.725,  0.843,  0.953),
    (0.500, 0.969,  0.969,  0.969),   # near-white centre
    (0.625, 0.984,  0.773,  0.627),
    (0.750, 0.918,  0.518,  0.345),
    (0.875, 0.753,  0.247,  0.157),
    (1.000, 0.502,  0.039,  0.067),   # deep red
]


def build_custom_cmap(name="custom_coolwarm"):
    """Build a LinearSegmentedColormap from the RGB node list above."""
    positions = [node[0] for node in CUSTOM_CMAP_RGB_NODES]
    r_vals    = [node[1] for node in CUSTOM_CMAP_RGB_NODES]
    g_vals    = [node[2] for node in CUSTOM_CMAP_RGB_NODES]
    b_vals    = [node[3] for node in CUSTOM_CMAP_RGB_NODES]

    cdict = {
        'red':   [(positions[i], r_vals[i], r_vals[i]) for i in range(len(positions))],
        'green': [(positions[i], g_vals[i], g_vals[i]) for i in range(len(positions))],
        'blue':  [(positions[i], b_vals[i], b_vals[i]) for i in range(len(positions))],
    }
    return LinearSegmentedColormap(name, cdict, N=256)


CMAP_OBJ = build_custom_cmap()


# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------
def load_nifti(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    img = nib.load(path)
    return img.get_fdata(), img


def get_middle_nonzero_slice(j_mag, plane='axial', threshold=0.0):
    if plane == 'axial':
        nonzero_slices = np.where(np.any(j_mag > threshold, axis=(0, 1)))[0]
        max_slices = j_mag.shape[2]
    elif plane == 'sagittal':
        nonzero_slices = np.where(np.any(j_mag > threshold, axis=(1, 2)))[0]
        max_slices = j_mag.shape[0]
    elif plane == 'coronal':
        nonzero_slices = np.where(np.any(j_mag > threshold, axis=(0, 2)))[0]
        max_slices = j_mag.shape[1]
    else:
        raise ValueError(f"Unknown plane: {plane}")

    if len(nonzero_slices) == 0:
        return max_slices // 2
    return int(nonzero_slices[len(nonzero_slices) // 2])


def spherical_to_cartesian(j_mag, theta, phi):
    if angles_are_degrees:
        theta = np.deg2rad(theta)
        phi   = np.deg2rad(phi)
    jx = j_mag * np.sin(theta) * np.cos(phi)
    jy = j_mag * np.sin(theta) * np.sin(phi)
    jz = j_mag * np.cos(theta)
    return jx, jy, jz


def smooth_slice(data, sigma=1.5):
    return gaussian_filter(data, sigma=sigma)


def smooth_mask_edges(mask, sigma=3.0, apply_morph=True, morph_iter=2):
    binary_mask = mask.astype(bool)
    if apply_morph:
        for _ in range(morph_iter):
            binary_mask = binary_closing(binary_mask)
    float_mask  = binary_mask.astype(float)
    smooth_mask = gaussian_filter(float_mask, sigma=sigma)
    if smooth_mask.max() > 0:
        smooth_mask = smooth_mask / smooth_mask.max()
    return smooth_mask


# ------------------------------------------------------------
# Input validation (debug aid): drop subjects that are missing files
# ------------------------------------------------------------
def validate_subjects(subject_ids, base_dir):
    valid = []
    for sid in subject_ids:
        subj_dir = os.path.join(base_dir, sid)
        needed   = [J_FILENAME, PHI_FILENAME, THETA_FILENAME]
        missing  = [f for f in needed if not os.path.exists(os.path.join(subj_dir, f))]
        if missing:
            log.warning("Skipping subject %s — missing file(s): %s", sid, ", ".join(missing))
        else:
            valid.append(sid)
    return valid


# ------------------------------------------------------------
# Slice-index resolution
# ------------------------------------------------------------
def resolve_slice_indices(subjects):
    indices = {}
    log.info("Determining slice indices...")
    for plane in plot_planes:
        if isinstance(slice_index, dict) and plane in slice_index:
            indices[plane] = slice_index[plane]
            log.info("  %-9s manual slice %d", plane.upper(), slice_index[plane])
        elif slice_index is not None and not isinstance(slice_index, dict):
            indices[plane] = slice_index
            log.info("  %-9s manual slice %d", plane.upper(), slice_index)
        else:
            first = subjects[0]
            jpath = os.path.join(base_dir, first, J_FILENAME)
            jm, _ = load_nifti(jpath)
            jm = np.nan_to_num(jm, nan=0.0, posinf=0.0, neginf=0.0)
            idx = get_middle_nonzero_slice(jm, plane, magnitude_threshold)
            indices[plane] = idx
            log.info("  %-9s auto-detected slice %d (from %s)", plane.upper(), idx, first)
    return indices


# ------------------------------------------------------------
# Global colour-limit computation (shared vmin / vmax for every figure)
# ------------------------------------------------------------
def compute_global_color_limits(subjects, base_dir, plane_slice_indices):
    log.info("Computing global colour limits...")
    all_valid_values = []

    for subject_id in subjects:
        j_path   = os.path.join(base_dir, subject_id, J_FILENAME)
        j_mag, _ = load_nifti(j_path)
        j_mag    = np.nan_to_num(j_mag, nan=0.0, posinf=0.0, neginf=0.0)

        for plane in plot_planes:
            slice_idx = plane_slice_indices[plane]

            if plane == 'axial':
                mag_slice = j_mag[:, :, slice_idx]
            elif plane == 'sagittal':
                mag_slice = j_mag[slice_idx, :, :]
            elif plane == 'coronal':
                mag_slice = j_mag[:, slice_idx, :]
            else:
                raise ValueError(f"Unknown plane: {plane}")

            mag_plot = np.rot90(mag_slice)
            if apply_smoothing:
                mag_plot = smooth_slice(mag_plot, sigma=gaussian_sigma)

            valid = mag_plot[mag_plot > magnitude_threshold]
            if len(valid) > 0:
                all_valid_values.extend(valid.tolist())

    if len(all_valid_values) == 0:
        log.warning("No values above threshold found; falling back to vmin=0, vmax=1.")
        return 0.0, 1.0

    all_valid_values = np.array(all_valid_values)
    vmin = float(np.percentile(all_valid_values, vmin_percentile))
    vmax = float(np.percentile(all_valid_values, vmax_percentile))
    return vmin, vmax


# ------------------------------------------------------------
# Slice computation (numeric pipeline — run ONCE per slice)
# ------------------------------------------------------------
@dataclass
class SliceRenderData:
    rgba: np.ndarray
    x_q: np.ndarray
    y_q: np.ndarray
    u_q: np.ndarray
    v_q: np.ndarray
    vmin: float
    vmax: float
    plane_label: str
    axis_label: str
    max_magnitude: float
    n_arrows: int


def compute_slice_render_data(j_mag, theta, phi, slice_idx, plane,
                              global_vmin, global_vmax):
    jx, jy, jz = spherical_to_cartesian(j_mag, theta, phi)

    if plane == 'axial':
        mag_slice   = j_mag[:, :, slice_idx]
        j1_slice    = jx[:, :, slice_idx]
        j2_slice    = jy[:, :, slice_idx]
        plane_label = "Axial (XY)"
        axis_label  = f"z = {slice_idx}"
    elif plane == 'sagittal':
        mag_slice   = j_mag[slice_idx, :, :]
        j1_slice    = jy[slice_idx, :, :]
        j2_slice    = jz[slice_idx, :, :]
        plane_label = "Sagittal (YZ)"
        axis_label  = f"x = {slice_idx}"
    elif plane == 'coronal':
        mag_slice   = j_mag[:, slice_idx, :]
        j1_slice    = jx[:, slice_idx, :]
        j2_slice    = jz[:, slice_idx, :]
        plane_label = "Coronal (XZ)"
        axis_label  = f"y = {slice_idx}"
    else:
        raise ValueError(f"Unknown plane: {plane}")

    mag_plot = np.rot90(mag_slice)
    j1_plot  = np.rot90(j1_slice)
    j2_plot  = np.rot90(j2_slice)

    # Smoothing
    if apply_smoothing:
        mag_plot_smooth = smooth_slice(mag_plot, sigma=gaussian_sigma)
        j1_plot_smooth  = smooth_slice(j1_plot,  sigma=gaussian_sigma)
        j2_plot_smooth  = smooth_slice(j2_plot,  sigma=gaussian_sigma)
    else:
        mag_plot_smooth = mag_plot
        j1_plot_smooth  = j1_plot
        j2_plot_smooth  = j2_plot

    # Mask & smooth edges
    mask_binary = mag_plot_smooth > magnitude_threshold
    mask_smooth = smooth_mask_edges(
        mask_binary,
        sigma=edge_smoothing_sigma,
        apply_morph=apply_morphology,
        morph_iter=morph_iterations,
    )
    alpha_channel = mask_smooth.copy()

    vmin = global_vmin
    vmax = global_vmax

    # Arrow grid
    ny, nx = mag_plot_smooth.shape
    y, x   = np.mgrid[0:ny, 0:nx]
    x_q    = x[::arrow_step, ::arrow_step]
    y_q    = y[::arrow_step, ::arrow_step]
    j1_q   = j1_plot_smooth[::arrow_step, ::arrow_step]
    j2_q   = j2_plot_smooth[::arrow_step, ::arrow_step]
    mask_q = mask_binary[::arrow_step, ::arrow_step]

    mag_values_nonzero = mag_plot_smooth[mask_binary]
    arrow_norm = np.percentile(mag_values_nonzero, 90) if len(mag_values_nonzero) > 0 else 1.0
    if arrow_norm == 0:
        arrow_norm = 1.0

    u_q =  j1_q / arrow_norm
    v_q = -j2_q / arrow_norm

    x_q = x_q[mask_q]; y_q = y_q[mask_q]
    u_q = u_q[mask_q]; v_q = v_q[mask_q]

    # Build RGBA image (guard against a zero-width colour range)
    denom = (vmax - vmin) if (vmax - vmin) != 0 else 1.0
    norm_mag = np.clip((mag_plot_smooth - vmin) / denom, 0, 1)
    rgba = CMAP_OBJ(norm_mag)
    rgba[:, :, 3] = alpha_channel

    return SliceRenderData(
        rgba=rgba, x_q=x_q, y_q=y_q, u_q=u_q, v_q=v_q,
        vmin=vmin, vmax=vmax,
        plane_label=plane_label, axis_label=axis_label,
        max_magnitude=float(np.max(mag_plot_smooth)),
        n_arrows=int(len(x_q)),
    )


# ------------------------------------------------------------
# Drawing (called once per FigureStyle — cheap, no re-computation)
# ------------------------------------------------------------
def draw_figure(render, subject_id, slice_idx, plane, out_subdir, style):
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(render.rgba, origin="upper", interpolation=interpolation, aspect='equal')

    norm_obj = Normalize(vmin=render.vmin, vmax=render.vmax)
    sm = cm.ScalarMappable(cmap=CMAP_OBJ, norm=norm_obj)
    sm.set_array([])

    if render.n_arrows > 0:
        ax.quiver(
            render.x_q, render.y_q, render.u_q, render.v_q,
            color=arrow_color,
            angles="xy", scale_units="xy", scale=arrow_scale,
            width=arrow_width, headwidth=5, headlength=6, headaxislength=5,
            alpha=arrow_alpha,
            edgecolors=arrow_edge_color, linewidth=arrow_edge_width,
            zorder=10,
        )

    # Colour bar
    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04, shrink=0.8)
    cbar.set_label(
        "Current density magnitude |J| (A/m²)",
        fontsize=style.cbar_label_font_size, fontweight=style.label_weight,
    )
    cbar.ax.tick_params(labelsize=style.cbar_tick_font_size)
    if style.tick_bold:
        for t in cbar.ax.get_yticklabels():
            t.set_fontweight("bold")

    # Title
    if style.simple_title:
        # e.g. "Axial (z = 40)", "Sagittal (x = 60)", "Coronal (y = 48)"
        title = f"{plane.capitalize()} ({render.axis_label})"
    else:
        title = (
            f"Participant {subject_id}: {render.plane_label} Current Density\n"
            f"Magnitude with In-plane Direction ({render.axis_label})"
        )
        if style.show_colorbar_range:
            title += f"\n[shared colorbar: {render.vmin:.4f} – {render.vmax:.4f} A/m²]"

    ax.set_title(title, fontsize=style.title_font_size,
                 fontweight=style.title_weight, pad=20)
    ax.axis("off")
    plt.tight_layout()

    stem = f"{subject_id}_{plane}_current_density_slice_{slice_idx}{style.filename_suffix}"
    png_path = os.path.join(out_subdir, stem + ".png")
    pdf_path = os.path.join(out_subdir, stem + ".pdf")
    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor='white')
    fig.savefig(pdf_path,           bbox_inches="tight", facecolor='white')
    plt.close(fig)

    tag = style.filename_suffix or "(standard)"
    log.info("    [%s %s] PNG %s", plane.upper(), tag, os.path.basename(png_path))
    log.debug("    [%s %s] PDF %s", plane.upper(), tag, pdf_path)
    return png_path, pdf_path


# ------------------------------------------------------------
# Misc reporting
# ------------------------------------------------------------
def print_colormap_nodes():
    log.info("Custom colormap RGB nodes (position, R, G, B) in [0, 1]:")
    for node in CUSTOM_CMAP_RGB_NODES:
        log.info("  pos=%.3f  R=%.3f  G=%.3f  B=%.3f", node[0], node[1], node[2], node[3])


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    log.info("=" * 60)
    log.info("Output directory: %s", output_dir)
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.isdir(base_dir):
        log.error("base_dir does not exist: %s", base_dir)
        return

    valid_subjects = validate_subjects(subject_ids, base_dir)
    if not valid_subjects:
        log.error("No valid subjects found — nothing to do.")
        return

    styles = build_styles()
    log.info("Figure groups per slice: %s",
             ", ".join(s.filename_suffix or "standard" for s in styles))

    # --- Step 1: slice indices ---
    plane_slice_indices = resolve_slice_indices(valid_subjects)

    # --- Step 2: shared colour limits across ALL subjects & planes ---
    global_vmin, global_vmax = compute_global_color_limits(
        valid_subjects, base_dir, plane_slice_indices
    )
    log.info("Global vmin (%sth pct): %.6f A/m²", vmin_percentile, global_vmin)
    log.info("Global vmax (%sth pct): %.6f A/m²", vmax_percentile, global_vmax)
    print_colormap_nodes()

    # --- Step 3: plot (with per-subject / per-plane failure isolation) ---
    failures = []
    for subject_id in valid_subjects:
        log.info("=" * 60)
        log.info("Processing subject %s ...", subject_id)

        try:
            subject_dir = os.path.join(base_dir, subject_id)
            j_mag, _ = load_nifti(os.path.join(subject_dir, J_FILENAME))
            phi,   _ = load_nifti(os.path.join(subject_dir, PHI_FILENAME))
            theta, _ = load_nifti(os.path.join(subject_dir, THETA_FILENAME))

            j_mag = np.nan_to_num(j_mag, nan=0.0, posinf=0.0, neginf=0.0)
            phi   = np.nan_to_num(phi,   nan=0.0, posinf=0.0, neginf=0.0)
            theta = np.nan_to_num(theta, nan=0.0, posinf=0.0, neginf=0.0)
        except Exception:
            log.error("Failed to load subject %s:\n%s", subject_id, traceback.format_exc())
            failures.append((subject_id, "load"))
            continue

        out_subdir = os.path.join(output_dir, subject_id)
        os.makedirs(out_subdir, exist_ok=True)

        for plane in plot_planes:
            slice_idx = plane_slice_indices[plane]
            try:
                log.info("  Plane %s (slice %d) ...", plane, slice_idx)
                render = compute_slice_render_data(
                    j_mag=j_mag, theta=theta, phi=phi,
                    slice_idx=slice_idx, plane=plane,
                    global_vmin=global_vmin, global_vmax=global_vmax,
                )
                log.info("    vmin=%.6f  vmax=%.6f  max|J|=%.6f  arrows=%d",
                         render.vmin, render.vmax, render.max_magnitude, render.n_arrows)

                for style in styles:
                    draw_figure(render, subject_id, slice_idx, plane, out_subdir, style)

            except Exception:
                log.error("Failed subject %s / plane %s:\n%s",
                          subject_id, plane, traceback.format_exc())
                failures.append((subject_id, plane))
                continue

    # --- Summary ---
    log.info("=" * 60)
    log.info("All plots finished.")
    log.info("  Planes: %s", ", ".join(plot_planes))
    log.info("  Slices: %s", plane_slice_indices)
    log.info("  Shared colorbar: [%.6f, %.6f] A/m²", global_vmin, global_vmax)
    log.info("  Figure groups: %s",
             ", ".join(s.filename_suffix or "standard" for s in styles))
    if failures:
        log.warning("  %d failure(s):", len(failures))
        for sid, what in failures:
            log.warning("    - subject %s (%s)", sid, what)
    else:
        log.info("  No failures.")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
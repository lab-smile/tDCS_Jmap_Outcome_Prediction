"""brainviz.py

Utilities to work with jMAP arrays and visualize brain images.

This module exposes a single public class, :class:`BrainVisualizer`, which
encapsulates two main capabilities:

1) Extracting the ``jmap_tp1`` column from a training dataframe into a NumPy
   array (``get_jmap_array``).
2) Visualizing 2D/3D/4D brain images with optional Harvard–Oxford ROI masking
   (``visualize_brain_image``).

The visualization logic mirrors the behavior in the original functions, with
additional safety checks, richer type hints, and a slightly more modular
structure. The class is stateless; instantiate it (optionally configuring the
atlas variant) and call its methods.

Example
-------
>>> import pandas as pd
>>> import numpy as np
>>> from brainviz import BrainVisualizer
>>>
>>> # Example dataframe with a nested list in 'jmap_tp1'
>>> X_train = pd.DataFrame({
...     'jmap_tp1': [np.random.randn(10, 10, 10).tolist()],
... }, index=[123])
>>>
>>> viz = BrainVisualizer()
>>> arr = viz.get_jmap_array(subject_id=123, X_train=X_train)
>>> saved = viz.visualize_brain_image(arr, save=False)  # show only, don't save

"""
from __future__ import annotations

import os
import ast
import datetime as _dt
from typing import Dict, Optional, Union, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from scipy.ndimage import gaussian_filter, binary_erosion

from nibabel import Nifti1Image, save as save_nii  # <-- add this import at top

# Optional neuroimaging stack
try:
    import nibabel as nib  # type: ignore
    from nilearn import datasets, image  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    nib = None  # type: ignore
    datasets = None  # type: ignore
    image = None  # type: ignore


__all__ = ["BrainVisualizer"]


class BrainVisualizer:
    """Helpers for extracting jMAP arrays and plotting brain images.

    Parameters
    ----------
    atlas_name : str, optional
        Harvard–Oxford atlas identifier to use when masking by ROI. Defaults to
        'cort-maxprob-thr25-2mm'.
    """

    def __init__(self, atlas_name: str = "cort-maxprob-thr25-2mm") -> None:
        self.atlas_name = atlas_name

    # ---------------------------------------------------------------------
    # Data helpers
    # ---------------------------------------------------------------------
    @staticmethod
    def get_jmap_array(subject_id: Union[int, str], X_train: pd.DataFrame) -> np.ndarray:
        """Return the ``jmap_tp1`` column as a NumPy array for a given subject.

        Parameters
        ----------
        subject_id : int or str
            Index value (``subjectid``) in the ``X_train`` DataFrame.
        X_train : pandas.DataFrame
            DataFrame with index = ``subjectid`` and a column ``'jmap_tp1'``
            containing nested lists (or any array-like) per subject.

        Returns
        -------
        numpy.ndarray
            Numpy array representation of the ``jmap_tp1`` data.
        """
        if "jmap_tp1" not in X_train.columns:
            raise KeyError("X_train must contain a 'jmap_tp1' column")
        try:
            value = X_train.loc[subject_id, "jmap_tp1"]
        except KeyError as e:  # re-raise with a clearer message
            raise KeyError(f"subject_id {subject_id!r} not found in X_train index") from e

        # Users sometimes store lists as strings; try literal_eval safely
        if isinstance(value, str):
            try:
                value = ast.literal_eval(value)
            except Exception:
                # Fall back to as-is; np.array will handle strings but warn user
                pass
        return np.array(value)

    # ---------------------------------------------------------------------
    # Visualization
    # ---------------------------------------------------------------------
    # ------------------------------------------------------------------
    # 3D multi-slice layouts
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # 3D multi-slice layouts
    # ------------------------------------------------------------------
    def visualize_brain_3d_grid(
        self,
        img: np.ndarray,
        plane: str = "axial",
        n_slices: int = 12,
        cols: int = 6,
        cmap: str = "turbo",
        out_dir: str = "jmap_visulize",
        save: bool = True,
        fname: Optional[str] = None,
        dpi: int = 600,
        smooth: bool = True,
        participant_id: Optional[Union[str, int]] = None,
        roi_name: Optional[str] = None,
        reference_img: Optional[Union[str, "nib.Nifti1Image"]] = None,
        smart_slicing: bool = True,
        # cropping + color scaling
        crop: bool = True,
        crop_pad: int = 2,
        crop_min_span: int = 16,
        robust_percentiles: Tuple[float, float] = (2.0, 98.0),
        # NEW: manual range override
        value_range: Optional[Tuple[float, float]] = None,
    ) -> Optional[Dict[str, str]]:

        # ---- Select data volume ----
        if img.ndim == 4:
            vol = img[..., 0]
        elif img.ndim == 3:
            vol = img
        else:
            raise ValueError("visualize_brain_3d_grid expects a 3D or 4D array")

        plane = plane.lower()
        if plane not in {"axial", "coronal", "sagittal"}:
            raise ValueError("plane must be one of {'axial','coronal','sagittal'}")

        # ---- Optional ROI masking ----
        title_suffix = f"  shape={tuple(vol.shape)}"
        if roi_name is not None:
            if reference_img is None:
                raise ValueError("`reference_img` is required when `roi_name` is provided.")
            ref_img = nib.load(reference_img) if isinstance(reference_img, str) else reference_img
            vol = self._mask_with_roi(vol, ref_img, roi_name)
            title_suffix += f"  [ROI: {roi_name}]"

        # ---- Axis & smart indices ----
        axis = {"axial": 2, "coronal": 1, "sagittal": 0}[plane]
        size = vol.shape[axis]

        def _smart_indices(v: np.ndarray, axis: int, k: int) -> np.ndarray:
            v_last = np.moveaxis(v, axis, -1)
            v_clean = np.nan_to_num(v_last, copy=False, nan=0)
            counts = np.count_nonzero(v_clean, axis=(0, 1)).astype(np.int64)
            if counts.sum() == 0:
                return np.linspace(0, v_clean.shape[-1]-1, k, dtype=int)
            cdf = np.cumsum(counts) / counts.sum()
            qs = (np.arange(k) + 0.5) / k
            idxs = np.searchsorted(cdf, qs, side="left").astype(int)
            idxs = np.clip(idxs, 0, v_clean.shape[-1]-1)
            uniq, seen = [], set()
            for i in idxs:
                if i not in seen:
                    uniq.append(i); seen.add(i)
            if len(uniq) < k:
                for c in np.linspace(0, v_clean.shape[-1]-1, k*3, dtype=int):
                    if c not in seen:
                        uniq.append(c); seen.add(c)
                        if len(uniq) == k: break
            return np.array(uniq[:k], dtype=int)

        idxs = _smart_indices(vol, axis, n_slices) if smart_slicing else np.linspace(0, size-1, n_slices, dtype=int)
        if smart_slicing: title_suffix += "  [smart slices]"

        # ---- Global x–y crop from non-zero distribution ----
        def _compute_xy_crop(v: np.ndarray, pad: int, min_span: int) -> Tuple[slice, slice, Tuple[int, int, int, int]]:
            m = (~np.isnan(v)) & (v != 0)
            if axis == 2:      proj = np.any(m, axis=2); x_len, y_len = v.shape[0], v.shape[1]
            elif axis == 1:    proj = np.any(m, axis=1); x_len, y_len = v.shape[0], v.shape[2]
            else:              proj = np.any(m, axis=0); x_len, y_len = v.shape[1], v.shape[2]
            x_any = np.any(proj, axis=1)
            y_any = np.any(proj, axis=0)

            def _bounds(any_vec, length):
                if not np.any(any_vec): return 0, length-1
                idxs = np.flatnonzero(any_vec); lo, hi = int(idxs[0]), int(idxs[-1])
                if hi - lo + 1 < min_span:
                    mid = (lo + hi) // 2; half = max(min_span // 2, 1)
                    lo, hi = mid - half, mid + half
                lo = max(lo - pad, 0); hi = min(hi + pad, length-1)
                return lo, hi

            x_lo, x_hi = _bounds(x_any, x_len)
            y_lo, y_hi = _bounds(y_any, y_len)
            return slice(x_lo, x_hi+1), slice(y_lo, y_hi+1), (x_lo, x_hi, y_lo, y_hi)

        if crop:
            xs, ys, (x_lo, x_hi, y_lo, y_hi) = _compute_xy_crop(vol, crop_pad, crop_min_span)
            title_suffix += f"  [crop x:{x_lo}-{x_hi}, y:{y_lo}-{y_hi}]"
        else:
            xs = slice(None); ys = slice(None)

        # ---- Shared vmin/vmax (manual override or robust percentiles) ----
        def _robust_limits(v: np.ndarray, xs: slice, ys: slice, pr: Tuple[float, float]) -> Tuple[float, float]:
            if axis == 2:      v_crop = v[xs, ys, :]
            elif axis == 1:    v_crop = v[xs, :, ys]
            else:              v_crop = v[:, xs, ys]
            arr = v_crop[np.isfinite(v_crop)]
            nz = arr[arr != 0]; base = nz if nz.size > 0 else arr
            if base.size == 0: return 0.0, 1.0
            p_lo, p_hi = np.percentile(base, [pr[0], pr[1]])
            if p_lo == p_hi:
                eps = 1e-6 if p_lo == 0 else abs(p_lo)*1e-6
                p_hi = p_lo + eps
            return float(p_lo), float(p_hi)

        if value_range is not None:
            vmin, vmax = value_range
            title_suffix += f"  [manual cbar {vmin:g}–{vmax:g}]"
        else:
            vmin, vmax = _robust_limits(vol, xs, ys, robust_percentiles)
            title_suffix += f"  [shared cbar {robust_percentiles[0]}–{robust_percentiles[1]}%]"

        # ---- Figure with dedicated colorbar column (no overlap) ----
        rows = int(np.ceil(n_slices / cols))
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
        fig = plt.figure(figsize=(2.5 * (cols + 0.4), 2.5 * rows))  # extra width for cbar
        gs = GridSpec(rows, cols + 1, figure=fig,
                    width_ratios=[1] * cols + [0.05],
                    wspace=0.08, hspace=0.12)

        axes = []
        for r in range(rows):
            for c in range(cols):
                axes.append(fig.add_subplot(gs[r, c]))
        cax = fig.add_subplot(gs[:, -1])  # dedicated colorbar axis

        # ---- Colormap + smoothing ----
        cm = plt.get_cmap(cmap)
        try:
            cm = cm.copy(); cm.set_bad(alpha=0.0)
        except Exception:
            if hasattr(cm, "with_extremes"):
                cm = cm.with_extremes(bad=(0, 0, 0, 0))
        interp = "bilinear" if smooth else "nearest"

        # ---- Draw slices ----
        images = []
        for i, ax in enumerate(axes):
            if i < len(idxs):
                k = int(idxs[i])
                if axis == 2:
                    sl = vol[:, :, k].T
                    sl = sl[ys, xs]
                elif axis == 1:
                    sl = vol[:, k, :].T
                    sl = sl[ys, xs]
                else:
                    sl = vol[k, :, :].T
                    sl = sl[ys, xs]
                im = ax.imshow(sl, cmap=cm, origin="lower",
                            interpolation=interp, vmin=vmin, vmax=vmax)
                images.append(im)
                ax.set_title(f"{plane.capitalize()} k={k}", fontsize=9)
            ax.axis("off")

        # ---- Unified colorbar in reserved axis ----
        if images:
            from matplotlib.colors import Normalize
            sm = plt.cm.ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap=cm)
            sm.set_array([])
            cb = fig.colorbar(sm, cax=cax)
            cb.ax.tick_params(labelsize=8)

        # ---- Title & layout (no overlap) ----
        pid_str = f"Participant {participant_id}" if participant_id is not None else "Participant NA"
        fig.suptitle(f"{plane.capitalize()} slices from {pid_str}{title_suffix}", y=0.98)
        fig.subplots_adjust(top=0.92, right=0.98)  # keep content clear of edges

        # ---- Save ----
        saved_paths: Optional[Dict[str, str]] = None
        if save:
            os.makedirs(out_dir, exist_ok=True)
            base_core = os.path.splitext(fname)[0] if fname else f"{plane}_grid_{_dt.datetime.now().strftime('%Y%m%d-%H%M%S')}"
            base = f"{participant_id}_" + base_core if participant_id is not None else base_core
            saved_paths = {}
            for ext in ("png", "svg", "pdf"):
                path = os.path.join(out_dir, f"{base}.{ext}")
                fig.savefig(path, dpi=dpi, bbox_inches="tight")
                saved_paths[ext] = path

        plt.show(); plt.close(fig)
        return saved_paths

    def visualize_brain_3d_cloud(
            self,
            img: np.ndarray,
            plane: str = "axial",
            n_slices: int = 12,
            cols: int = 6,  # kept for signature parity (unused)
            cmap: str = "turbo",
            out_dir: str = "jmap_visulize",
            save: bool = True,
            fname: Optional[str] = None,
            dpi: int = 600,
            smooth: bool = False,
            participant_id: Optional[Union[str, int]] = None,
            roi_name: Optional[str] = None,
            reference_img: Optional[Union[str, "nib.Nifti1Image"]] = None,
            smart_slicing: bool = True,
            # cropping + color scaling
            crop: bool = True,
            crop_pad: int = 2,
            crop_min_span: int = 16,
            robust_percentiles: Tuple[float, float] = (2.0, 98.0),  # kept for API parity (unused if value_range=None)
            # manual range override
            value_range: Optional[Tuple[float, float]] = None,
        ) -> Optional[Dict[str, str]]:
        """
        Dense 3D volume rendering (VTK ray casting) of the selected region,
        showing ALL voxels inside the ROI (including zeros). Background is ONLY NaN.
        If `value_range` is None, the colormap range is the true [min,max] of ROI data.
        """
        import os
        import datetime as _dt
        import numpy as np
        import nibabel as nib
        import matplotlib
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize, ListedColormap
        import pyvista as pv

        # ---------------- Select data volume ----------------
        if img.ndim == 4:
            vol = img[..., 0]
        elif img.ndim == 3:
            vol = img
        else:
            raise ValueError("visualize_brain_3d_volume expects a 3D or 4D array")

        plane = plane.lower()
        if plane not in {"axial", "coronal", "sagittal"}:
            raise ValueError("plane must be one of {'axial','coronal','sagittal'}")

        title_suffix = f"  shape={tuple(vol.shape)}"
        # -------- Apply ROI (required for 'show all voxels in selected ROI') --------
        if roi_name is not None:
            if reference_img is None:
                raise ValueError("`reference_img` is required when `roi_name` is provided.")
            ref_img = nib.load(reference_img) if isinstance(reference_img, str) else reference_img
            # Expectation: _mask_with_roi returns a volume where voxels OUTSIDE ROI are set to NaN
            # If your implementation sets outside-ROI to 0, consider changing it to NaN there.
            vol = self._mask_with_roi(vol, ref_img, roi_name)
            title_suffix += f"  [ROI: {roi_name}]"

        axis = {"axial": 2, "coronal": 1, "sagittal": 0}[plane]
        size = vol.shape[axis]

        # ---------------- Smart slice selection (based on finite voxels, not nonzero) ----------------
        def _smart_indices(v: np.ndarray, axis: int, k: int) -> np.ndarray:
            v_last = np.moveaxis(v, axis, -1)
            v_finite = np.isfinite(v_last)
            counts = np.count_nonzero(v_finite, axis=(0, 1)).astype(np.int64)
            if counts.sum() == 0:
                return np.linspace(0, v_last.shape[-1]-1, k, dtype=int)
            cdf = np.cumsum(counts) / counts.sum()
            qs = (np.arange(k) + 0.5) / k
            idxs = np.searchsorted(cdf, qs, side="left").astype(int)
            idxs = np.clip(idxs, 0, v_last.shape[-1]-1)
            uniq, seen = [], set()
            for i in idxs:
                if i not in seen:
                    uniq.append(i); seen.add(i)
            if len(uniq) < k:
                for c in np.linspace(0, v_last.shape[-1]-1, k*3, dtype=int):
                    if c not in seen:
                        uniq.append(c); seen.add(c)
                        if len(uniq) == k: break
            return np.array(uniq[:k], dtype=int)

        idxs = _smart_indices(vol, axis, n_slices) if smart_slicing else np.linspace(0, size-1, n_slices, dtype=int)
        if smart_slicing:
            title_suffix += "  [smart slices]"

        # ---------------- XY crop (based on finite voxels so zeros inside ROI are kept) ----------------
        def _compute_xy_crop(v: np.ndarray, pad: int, min_span: int):
            m = np.isfinite(v)  # include zeros inside ROI; exclude NaNs (outside ROI)
            if axis == 2:
                proj = np.any(m, axis=2); x_len, y_len = v.shape[0], v.shape[1]
            elif axis == 1:
                proj = np.any(m, axis=1); x_len, y_len = v.shape[0], v.shape[2]
            else:
                proj = np.any(m, axis=0); x_len, y_len = v.shape[1], v.shape[2]
            x_any = np.any(proj, axis=1)
            y_any = np.any(proj, axis=0)

            def _bounds(any_vec, length):
                if not np.any(any_vec): return 0, length-1
                idxs_b = np.flatnonzero(any_vec); lo, hi = int(idxs_b[0]), int(idxs_b[-1])
                if hi - lo + 1 < min_span:
                    mid = (lo + hi) // 2; half = max(min_span // 2, 1)
                    lo, hi = mid - half, mid + half
                lo = max(lo - pad, 0); hi = min(hi + pad, length-1)
                return lo, hi

            x_lo, x_hi = _bounds(x_any, x_len)
            y_lo, y_hi = _bounds(y_any, y_len)
            return slice(x_lo, x_hi+1), slice(y_lo, y_hi+1), (x_lo, x_hi, y_lo, y_hi)

        if crop:
            xs, ys, (x_lo, x_hi, y_lo, y_hi) = _compute_xy_crop(vol, crop_pad, crop_min_span)
            title_suffix += f"  [crop x:{x_lo}-{x_hi}, y:{y_lo}-{y_hi}]"
        else:
            xs = slice(None); ys = slice(None)

        # ---------------- Determine color scaling ----------------
        # For the "show ALL voxels" requirement: when value_range is None, use TRUE [min,max] of finite ROI data.
        def _true_limits(v: np.ndarray, xs: slice, ys: slice):
            if axis == 2:      v_crop = v[xs, ys, :]
            elif axis == 1:    v_crop = v[xs, :, ys]
            else:              v_crop = v[:, xs, ys]
            arr = v_crop[np.isfinite(v_crop)]
            if arr.size == 0:
                return 0.0, 1.0
            lo, hi = float(np.min(arr)), float(np.max(arr))
            if lo == hi:
                eps = 1e-6 if lo == 0 else abs(lo)*1e-6
                hi = lo + eps
            return lo, hi

        if value_range is not None:
            vmin, vmax = value_range
            title_suffix += f"  [manual cbar {vmin:g}–{vmax:g}]"
        else:
            vmin, vmax = _true_limits(vol, xs, ys)
            title_suffix += "  [cbar=min–max of ROI]"

        # ---------------- Extract compact subvolume spanning chosen slices ----------------
        all_k = idxs.astype(int)
        kmin, kmax = int(all_k.min()), int(all_k.max())
        if axis == 2:      sub = vol[xs, ys, kmin:kmax+1]
        elif axis == 1:    sub = vol[xs, kmin:kmax+1, ys]
        else:              sub = vol[kmin:kmax+1, xs, ys]
        sub = sub.astype(np.float32, copy=True)

        # ======== Background handling: ONLY NaNs are background (zeros are valid voxels) ========
        bg_mask = ~np.isfinite(sub)

        # ======== Optional smoothing (NaN-aware, preserves background exactly) ========
        if smooth:
            sigma = 0.8

            # Original valid-data mask (True where values are valid)
            val_mask = (~bg_mask) & np.isfinite(sub)

            # Smooth numerator and mask, then normalize
            num = gaussian_filter(np.nan_to_num(sub, nan=0.0) * val_mask.astype(np.float32),
                                sigma=sigma, mode="reflect")
            den = gaussian_filter(val_mask.astype(np.float32),
                                sigma=sigma, mode="reflect")

            with np.errstate(invalid="ignore", divide="ignore"):
                smoothed = num / np.maximum(den, 1e-8)

            # Critically: re-apply the original mask so support doesn't grow
            sub = np.where(val_mask, smoothed, np.nan)

            # Keep bg_mask in sync
            bg_mask = ~np.isfinite(sub)
        # Replace NaNs with a sentinel just below vmin so they map outside the visible range
        rng = max(1e-6, (vmax - vmin))
        sentinel = vmin - 0.05 * rng
        sub_bg_filled = sub.copy()
        sub_bg_filled[bg_mask] = sentinel
        #This keeps outside-ROI voxels truly transparent and removes the “shadow.”
        sub_bg_filled[bg_mask] = np.nan   # <-- NaNs mark "pure background"


        # ---------------- Build PyVista grid ----------------
        grid = pv.ImageData()
        grid.dimensions = np.array(sub_bg_filled.shape)
        grid.spacing = (1.0, 1.0, 1.0)
        grid.origin = (0.0, 0.0, 0.0)
        grid.point_data["scalars"] = np.ascontiguousarray(sub_bg_filled).ravel(order="F")

        # --- Build ROI boundary shell for crisp outline ---
        finite = np.isfinite(sub_bg_filled)   # inside ROI = True
        mask_grid = pv.ImageData()
        mask_grid.dimensions = np.array(sub_bg_filled.shape)
        mask_grid.spacing = (1.0, 1.0, 1.0)
        mask_grid.origin = (0.0, 0.0, 0.0)
        mask_grid.point_data["m"] = finite.astype(np.uint8).ravel(order="F")

        # Extract ROI boundary surface
        roi_shell = mask_grid.contour([0.5], scalars="m")  # 0/1 boundary


        
        # --- remove NaN voxels completely so they are invisible ---
        finite = np.isfinite(sub_bg_filled).ravel(order="F")
        grid.point_data["mask"] = finite.astype(np.uint8)
        grid = grid.threshold(0.5, scalars="mask", preference="point")

        # ---------------- Colormap (turbo by default) ----------------
        base_cmap = matplotlib.colormaps.get(cmap)
        num_bins = 64
        bin_edges = np.linspace(vmin, vmax, num_bins + 1)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        colors_disc = base_cmap(Normalize(vmin=vmin, vmax=vmax)(bin_centers))
        listed = ListedColormap(colors_disc)

        # ---------------- Opacity transfer function ----------------
        # No "dead zone": every in-range value gets some opacity so zeros are visible.
        opacity_n = 128
        opac = np.linspace(0.05, 1.0, opacity_n) ** 1.1  # gentle ramp; small floor so faint values show
        opac = (np.linspace(0.0, 1.0, opacity_n) ** 1.1)
        opac[0] = 0.0

        # ---------------- Headless-safe rendering ----------------
        try:
            pv.start_xvfb()
        except Exception:
            pass

        saved_paths: Optional[Dict[str, str]] = None
        if save:
            os.makedirs(out_dir, exist_ok=True)

        p = pv.Plotter(off_screen=True, window_size=(1200, 900))
        p.set_background("white")

        p.add_volume(
            grid,
            scalars="scalars",
            cmap=listed,
            opacity=opac,
            clim=(vmin, vmax),
            blending="composite",
            shade=False,
            scalar_bar_args=dict(
                title="Current density\n(A/m²)",   # <-- custom title
                vertical=True,
                n_labels=7,
                title_font_size=18,
                label_font_size=14,
                fmt="%.4g",
                position_x=0.90,
                position_y=0.10,
            ),
        )

        # p.add_mesh(roi_shell, color="grey", opacity=0.7, smooth_shading=True)

        # Camera
        p.camera.azimuth = -60
        p.camera.elevation = 20
        p.camera.zoom(1.2)

        base_core = os.path.splitext(fname)[0] if fname else f"{plane}_volume_{_dt.datetime.now().strftime('%Y%m%d-%H%M%S')}"
        base = f"{participant_id}_" + base_core if participant_id is not None else base_core
        stem = os.path.join(out_dir, base)

        png_path = stem + ".png"
        svg_path = stem + ".svg"
        pdf_path = stem + ".pdf"

        # 1) Render but DON'T auto-close
        #    (use off_screen=True when constructing the Plotter if you're on a headless machine)
        p.show(auto_close=False)

        # 2) Export vector graphics (requires VTK with GL2PS)
        p.save_graphic(svg_path)
        p.save_graphic(pdf_path)

        # 3) Save raster screenshot
        p.screenshot(png_path)

        # 4) Now it's safe to close
        p.close()

        # ---------------- Histogram (same bins/colors, exclude NaN/sentinel) ----------------
        arr = sub[np.isfinite(sub)]
        in_rng = (arr >= vmin) & (arr <= vmax)
        data = arr[in_rng]
        counts, _ = np.histogram(data, bins=bin_edges)
        widths = np.diff(bin_edges)

        fig, ax = plt.subplots(figsize=(8, 3))
        ax.bar(bin_edges[:-1], counts, width=widths, align="edge",
            color=colors_disc, edgecolor="black", linewidth=0.3)
        ax.set_xlim(vmin, vmax)
        ax.set_xlabel("Voxel value")
        ax.set_ylabel("Count")
        ax.grid(True, linestyle="--", alpha=0.25)
        if counts.sum() > 0:
            mean_v = float(np.mean(data)); med_v = float(np.median(data))
            ax.axvline(mean_v, linestyle="--", linewidth=1.0, color="k")
            ax.axvline(med_v, linestyle=":",  linewidth=1.0, color="k")
            ax.legend([f"mean={mean_v:.3g}", f"median={med_v:.3g}"], frameon=False, fontsize=8)
        ax.set_title(f"Histogram (bins match volume colormap)  range=[{vmin:.3g}, {vmax:.3g}]")
        # Define base output path (without extension)
        hist_path = os.path.join(out_dir, base + "_hist")

        # Save in multiple formats
        fig.savefig(f"{hist_path}.png", dpi=dpi, bbox_inches="tight")
        fig.savefig(f"{hist_path}.svg", bbox_inches="tight")
        fig.savefig(f"{hist_path}.pdf", bbox_inches="tight")
        plt.close(fig)
        saved_paths = {"volume_png": png_path, "volume_svg": svg_path, "volume_pdf": pdf_path, "histogram_png": hist_path}

        return saved_paths



    def visualize_brain_3d_histogram(
        self,
        img: np.ndarray,
        plane: str = "axial",
        n_slices: int = 12,
        cols: int = 6,                     # unused here but kept for signature parity
        cmap: str = "turbo",
        out_dir: str = "jmap_visulize",
        save: bool = True,
        fname: Optional[str] = None,
        dpi: int = 600,
        smooth: bool = True,               # unused for histogram but kept for signature parity
        participant_id: Optional[Union[str, int]] = None,
        roi_name: Optional[str] = None,
        reference_img: Optional[Union[str, "nib.Nifti1Image"]] = None,
        smart_slicing: bool = True,
        # cropping + color scaling
        crop: bool = True,
        crop_pad: int = 2,
        crop_min_span: int = 16,
        robust_percentiles: Tuple[float, float] = (2.0, 98.0),
        # manual range override
        value_range: Optional[Tuple[float, float]] = None,
    ) -> Optional[Dict[str, str]]:
        """
        Render a histogram of voxel intensities from the selected region/slices,
        with bar colors matched to the colormap and a corresponding colorbar.
        Uses the same selection logic (ROI, cropping, smart slicing) and the
        same value range determination as visualize_brain_3d_grid.
        """
        import os
        import datetime as _dt
        import numpy as np
        import nibabel as nib
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize

        # ---- Select data volume ----
        if img.ndim == 4:
            vol = img[..., 0]
        elif img.ndim == 3:
            vol = img
        else:
            raise ValueError("visualize_brain_3d_histogram expects a 3D or 4D array")

        plane = plane.lower()
        if plane not in {"axial", "coronal", "sagittal"}:
            raise ValueError("plane must be one of {'axial','coronal','sagittal'}")

        # ---- Optional ROI masking ----
        title_suffix = f"  shape={tuple(vol.shape)}"
        if roi_name is not None:
            if reference_img is None:
                raise ValueError("`reference_img` is required when `roi_name` is provided.")
            ref_img = nib.load(reference_img) if isinstance(reference_img, str) else reference_img
            vol = self._mask_with_roi(vol, ref_img, roi_name)
            title_suffix += f"  [ROI: {roi_name}]"

        # ---- Axis & smart indices ----
        axis = {"axial": 2, "coronal": 1, "sagittal": 0}[plane]
        size = vol.shape[axis]

        def _smart_indices(v: np.ndarray, axis: int, k: int) -> np.ndarray:
            v_last = np.moveaxis(v, axis, -1)
            v_clean = np.nan_to_num(v_last, copy=False, nan=0)
            counts = np.count_nonzero(v_clean, axis=(0, 1)).astype(np.int64)
            if counts.sum() == 0:
                return np.linspace(0, v_clean.shape[-1]-1, k, dtype=int)
            cdf = np.cumsum(counts) / counts.sum()
            qs = (np.arange(k) + 0.5) / k
            idxs = np.searchsorted(cdf, qs, side="left").astype(int)
            idxs = np.clip(idxs, 0, v_clean.shape[-1]-1)
            uniq, seen = [], set()
            for i in idxs:
                if i not in seen:
                    uniq.append(i); seen.add(i)
            if len(uniq) < k:
                for c in np.linspace(0, v_clean.shape[-1]-1, k*3, dtype=int):
                    if c not in seen:
                        uniq.append(c); seen.add(c)
                        if len(uniq) == k: break
            return np.array(uniq[:k], dtype=int)

        idxs = _smart_indices(vol, axis, n_slices) if smart_slicing else np.linspace(0, size-1, n_slices, dtype=int)
        if smart_slicing: title_suffix += "  [smart slices]"

        # ---- Global x–y crop from non-zero distribution ----
        def _compute_xy_crop(v: np.ndarray, pad: int, min_span: int) -> Tuple[slice, slice, Tuple[int, int, int, int]]:
            m = (~np.isnan(v)) & (v != 0)
            if axis == 2:      proj = np.any(m, axis=2); x_len, y_len = v.shape[0], v.shape[1]
            elif axis == 1:    proj = np.any(m, axis=1); x_len, y_len = v.shape[0], v.shape[2]
            else:              proj = np.any(m, axis=0); x_len, y_len = v.shape[1], v.shape[2]
            x_any = np.any(proj, axis=1)
            y_any = np.any(proj, axis=0)

            def _bounds(any_vec, length):
                if not np.any(any_vec): return 0, length-1
                idxs_b = np.flatnonzero(any_vec); lo, hi = int(idxs_b[0]), int(idxs_b[-1])
                if hi - lo + 1 < min_span:
                    mid = (lo + hi) // 2; half = max(min_span // 2, 1)
                    lo, hi = mid - half, mid + half
                lo = max(lo - pad, 0); hi = min(hi + pad, length-1)
                return lo, hi

            x_lo, x_hi = _bounds(x_any, x_len)
            y_lo, y_hi = _bounds(y_any, y_len)
            return slice(x_lo, x_hi+1), slice(y_lo, y_hi+1), (x_lo, x_hi, y_lo, y_hi)

        if crop:
            xs, ys, (x_lo, x_hi, y_lo, y_hi) = _compute_xy_crop(vol, crop_pad, crop_min_span)
            title_suffix += f"  [crop x:{x_lo}-{x_hi}, y:{y_lo}-{y_hi}]"
        else:
            xs = slice(None); ys = slice(None)

        # ---- Shared vmin/vmax (manual override or robust percentiles) ----
        def _robust_limits(v: np.ndarray, xs: slice, ys: slice, pr: Tuple[float, float]) -> Tuple[float, float]:
            if axis == 2:      v_crop = v[xs, ys, :]
            elif axis == 1:    v_crop = v[xs, :, ys]
            else:              v_crop = v[:, xs, ys]
            arr = v_crop[np.isfinite(v_crop)]
            nz = arr[arr != 0]; base = nz if nz.size > 0 else arr
            if base.size == 0: return 0.0, 1.0
            p_lo, p_hi = np.percentile(base, [pr[0], pr[1]])
            if p_lo == p_hi:
                eps = 1e-6 if p_lo == 0 else abs(p_lo)*1e-6
                p_hi = p_lo + eps
            return float(p_lo), float(p_hi)

        if value_range is not None:
            vmin, vmax = value_range
            title_suffix += f"  [manual cbar {vmin:g}–{vmax:g}]"
        else:
            vmin, vmax = _robust_limits(vol, xs, ys, robust_percentiles)
            title_suffix += f"  [shared cbar {robust_percentiles[0]}–{robust_percentiles[1]}%]"

        # ---- Gather voxel values from selected slices & region ----
        vals = []
        for k in idxs:
            k = int(k)
            if axis == 2:
                sl = vol[:, :, k].T
                sl = sl[ys, xs]
            elif axis == 1:
                sl = vol[:, k, :].T
                sl = sl[ys, xs]
            else:
                sl = vol[k, :, :].T
                sl = sl[ys, xs]
            vals.append(sl.reshape(-1))
        if len(vals) == 0:
            raise ValueError("No slices selected for histogram.")
        vals = np.concatenate(vals, axis=0)

        # Finite & optionally exclude zeros (mirror robust logic)
        is_finite = np.isfinite(vals)
        nz = vals[is_finite & (vals != 0)]
        base = nz if nz.size > 0 else vals[is_finite]

        # Restrict to [vmin, vmax] for binning (matches displayed range)
        in_range = (base >= vmin) & (base <= vmax)
        data = base[in_range]
        total_count = data.size

        # ---- Build histogram with bin colors tied to cmap/colorbar ----
        # Keep signature identical; choose a sensible fixed bin count
        num_bins = 64
        bin_edges = np.linspace(vmin, vmax, num_bins + 1)
        counts, edges = np.histogram(data, bins=bin_edges)

        # Bin centers for coloring
        centers = 0.5 * (edges[:-1] + edges[1:])
        norm = Normalize(vmin=vmin, vmax=vmax)
        cm = plt.get_cmap(cmap)
        try:
            cm = cm.copy(); cm.set_bad(alpha=0.0)
        except Exception:
            if hasattr(cm, "with_extremes"):
                cm = cm.with_extremes(bad=(0, 0, 0, 0))

        # ---- Figure with dedicated colorbar axis ----
        fig = plt.figure(figsize=(8.5, 4.8))
        gs = fig.add_gridspec(1, 2, width_ratios=[0.92, 0.08], wspace=0.15)
        ax_hist = fig.add_subplot(gs[0, 0])
        cax = fig.add_subplot(gs[0, 1])

        # Draw colored bars (vivid histogram)
        widths = np.diff(edges)
        colors = cm(norm(centers))
        bars = ax_hist.bar(edges[:-1], counts, width=widths, align="edge",
                        color=colors, edgecolor="black", linewidth=0.3)

        ax_hist.set_xlim(vmin, vmax)
        ax_hist.set_xlabel("Voxel value")
        ax_hist.set_ylabel("Count")
        ax_hist.grid(True, alpha=0.25, linestyle="--")

        # Stats annotations
        if total_count > 0:
            mean_v = float(np.mean(data))
            med_v = float(np.median(data))
            ax_hist.axvline(mean_v, linestyle="--", linewidth=1.0, color="k")
            ax_hist.axvline(med_v, linestyle=":", linewidth=1.0, color="k")
            ax_hist.legend([f"mean={mean_v:.3g}", f"median={med_v:.3g}"], frameon=False)

        # Colorbar matching the histogram/cmap
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cm)
        sm.set_array([])
        cb = fig.colorbar(sm, cax=cax)
        cb.ax.tick_params(labelsize=8)

        # Title
        pid_str = f"Participant {participant_id}" if participant_id is not None else "Participant NA"
        fig.suptitle(f"Value distribution ({plane} slices) from {pid_str}{title_suffix}\n"
                    f"n_slices={len(idxs)}  voxels={total_count}", y=0.98)

        fig.subplots_adjust(top=0.88, right=0.98)

        # ---- Save ----
        saved_paths: Optional[Dict[str, str]] = None
        if save:
            os.makedirs(out_dir, exist_ok=True)
            base_core = os.path.splitext(fname)[0] if fname else f"{plane}_hist_{_dt.datetime.now().strftime('%Y%m%d-%H%M%S')}"
            base = f"{participant_id}_" + base_core if participant_id is not None else base_core
            saved_paths = {}
            for ext in ("png", "svg", "pdf"):
                path = os.path.join(out_dir, f"{base}.{ext}")
                fig.savefig(path, dpi=dpi, bbox_inches="tight")
                saved_paths[ext] = path

        plt.show(); plt.close(fig)
        return saved_paths


    
    def visualize_brain_image(
        self,
        img: np.ndarray,
        cmap: str = "turbo",
        save: bool = True,
        out_dir: str = "jmap_visulize",
        fname: Optional[str] = None,
        dpi: int = 600,
        close: bool = True,
        downsample: Optional[float] = None,
        smooth: bool = True,
        roi_name: Optional[str] = None,
        reference_img: Optional[Union[str, "nib.Nifti1Image"]] = None,
        list_rois: bool = False,
        participant_id: Optional[Union[str, int]] = None,
    ) -> Optional[Dict[str, str]]:
        """Visualize a brain image (2D/3D/4D) and optionally save it.

        Can optionally mask to a single Harvard–Oxford ROI (like plot_shap_from_csv).

        Parameters
        ----------
        img : numpy.ndarray
            Brain image data. Supports 2D, 3D, or 4D (time) arrays.
        cmap : str, optional
            Matplotlib colormap. Default 'turbo'.
        save : bool, optional
            If True, saves the figure(s) to ``out_dir`` as PNG/SVG/PDF. Default True.
        out_dir : str, optional
            Directory to save images. Will be created if it doesn't exist.
        fname : str | None, optional
            Base filename (without extension). If None, a name is auto-generated.
        dpi : int, optional
            DPI for raster outputs (PNG, PDF). Default 600.
        close : bool, optional
            If True, closes the figure after saving/showing to free memory. Default True.
        downsample : float | None, optional
            Factor to resample the image for visualization:
            - None or 1 → No downsampling (voxel-by-voxel).
            - >1 → Downsample by this factor (e.g., 2 → half resolution).
            - 0<factor<1 → Upsample (e.g., 0.5 → double resolution).
        smooth : bool, optional
            If True, smooths the displayed figure (bilinear). If False, uses nearest.
        roi_name : str | None, optional
            Name of Harvard–Oxford ROI to display (case-insensitive). If provided,
            the function masks out all other voxels. Requires ``reference_img``.
        reference_img : nib.Nifti1Image | str | None, optional
            NIfTI image (or path) defining the target grid/affine for atlas resampling.
            Required if ``roi_name`` is set.
        list_rois : bool, optional
            If True, prints and returns the available ROI labels (no plotting).

        Returns
        -------
        dict | None
            If ``save=True``, returns a dict with saved file paths. Otherwise None.
            If ``list_rois=True``, returns the saved paths if plotting occurred, or
            ``None`` if only ROI listing was requested.
        """
        # ROI listing only
        if list_rois:
            self._print_available_rois()

        # ---- Select data volume ----
        if img.ndim == 4:
            vol = img[..., 0]
            title_suffix = f" (t=0)  shape={tuple(vol.shape)}"
        elif img.ndim in (2, 3):
            vol = img
            title_suffix = f"  shape={tuple(vol.shape)}"
        else:
            raise ValueError(f"Unsupported array dimensions: {img.ndim}")

        # ---- Optional down/up-sample for visualization ----
        if downsample is not None and downsample != 1:
            if downsample <= 0:
                raise ValueError("downsample factor must be > 0.")
            vol = zoom(vol, [1 / downsample] * vol.ndim, order=1)  # bilinear
            title_suffix += f"  [downsample={downsample}x]"

        # ---- Optional ROI masking via Harvard–Oxford ----
        if roi_name is not None:
            if vol.ndim != 3:
                raise ValueError("ROI selection requires a 3D volume.")
            if reference_img is None:
                raise ValueError("`reference_img` is required when `roi_name` is provided.")
            if nib is None or datasets is None or image is None:
                raise ImportError(
                    "nibabel and nilearn must be installed to use ROI masking."
                )

            ref_img = nib.load(reference_img) if isinstance(reference_img, str) else reference_img
            vol = self._mask_with_roi(vol, ref_img, roi_name)
            title_suffix += f"  [ROI: {roi_name}]"

        # ---- Colormap + smoothing ----
        interp_method = "bilinear" if smooth else "nearest"
        cm = plt.get_cmap(cmap)
        # Make NaNs transparent (outside-ROI becomes invisible) if supported
        try:
            cm = cm.copy()
            cm.set_bad(alpha=0.0)
        except Exception:
            if hasattr(cm, "with_extremes"):
                cm = cm.with_extremes(bad=(0, 0, 0, 0))

        # ---- Plot ----
        saved_paths: Optional[Dict[str, str]] = None
        if vol.ndim == 2:
            fig = plt.figure(figsize=(5, 5))
            plt.imshow(vol, cmap=cm, interpolation=interp_method)
            plt.title("2D Brain Slice" + title_suffix)
            plt.axis("off")
        else:
            z_mid = vol.shape[2] // 2
            y_mid = vol.shape[1] // 2
            x_mid = vol.shape[0] // 2

            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            axes[0].imshow(vol[:, :, z_mid].T, cmap=cm, origin="lower", interpolation=interp_method)
            axes[0].set_title(f"Axial (z={z_mid})")
            axes[1].imshow(vol[:, y_mid, :].T, cmap=cm, origin="lower", interpolation=interp_method)
            axes[1].set_title(f"Coronal (y={y_mid})")
            axes[2].imshow(vol[x_mid, :, :].T, cmap=cm, origin="lower", interpolation=interp_method)
            axes[2].set_title(f"Sagittal (x={x_mid})")
            for ax in axes:
                ax.axis("off")
            fig.suptitle("3D Brain (mid-slices)" + title_suffix, y=0.98)

        plt.tight_layout()

        # ---- Save ----
        if save:
            os.makedirs(out_dir, exist_ok=True)
            base = os.path.splitext(fname)[0] if fname else f"{str(participant_id)}_brain_vis_{_dt.datetime.now().strftime('%Y%m%d-%H%M%S')}"
            saved_paths = {}
            for ext in ("png", "svg", "pdf"):
                path = os.path.join(out_dir, f"{base}.{ext}")
                fig.savefig(path, dpi=dpi, bbox_inches="tight")
                saved_paths[ext] = path
            # save as NIfTI (.nii.gz)
            BrainVisualizer.save_as_nii(vol, reference_img, out_dir, base, saved_paths)


        plt.show()
        if close:
            plt.close(fig)

        return saved_paths

    @staticmethod
    def save_as_nii( vol, reference_img, out_dir, base, saved_paths):
        # >>> NEW: save as NIfTI (.nii.gz)
        if vol.ndim == 3:
            # use reference affine if provided; otherwise identity
            if isinstance(reference_img, str):
                ref_img_obj = nib.load(reference_img)
                affine = ref_img_obj.affine
            elif reference_img is not None:
                affine = reference_img.affine
            else:
                affine = np.eye(4)

            nii_img = Nifti1Image(vol.astype(np.float32), affine)
            nii_path = os.path.join(out_dir, f"{base}.nii.gz")
            save_nii(nii_img, nii_path)
            saved_paths["nii"] = nii_path
        else:
            print('[BrainVisualizer] vol.ndim != 3, check data shape')
        return saved_paths

    @staticmethod
    def _stream_numeric(x):
        """Yield numeric floats from arbitrary nested structures (ignores NaN)."""
        if isinstance(x, np.ndarray):
            if np.issubdtype(x.dtype, np.number):
                for v in x.ravel(order="K"):
                    if np.isfinite(v):
                        yield float(v)
            else:
                for elem in x.flat:
                    yield from BrainVisualizer._stream_numeric(elem)
            return
        if isinstance(x, numbers.Number) or isinstance(x, (np.generic,)):
            if np.isfinite(x):
                yield float(x)
            return
        if isinstance(x, str):
            try:
                v = float(x)
                if np.isfinite(v):
                    yield v
                return
            except Exception:
                try:
                    parsed = ast.literal_eval(x)
                except Exception:
                    return
                yield from BrainVisualizer._stream_numeric(parsed)
            return
        if isinstance(x, (list, tuple)):
            for elem in x:
                yield from BrainVisualizer._stream_numeric(elem)
            return

    def compute_value_range(self,
                            X_train,
                            roi_name: str = None,
                            reference_img=None):
        """
        Compute a GLOBAL (vmin, vmax) across all subjects' jmap_tp1 volumes.
        If roi_name/reference_img are provided, restrict the min/max to that ROI.

        Parameters
        ----------
        X_train : pandas.DataFrame
            Must contain a 'jmap_tp1' column and have subject IDs as index.
        roi_name : str | None
            Harvard–Oxford ROI name (case-insensitive). If provided, each subject’s
            array is masked to this ROI before min/max are computed.
        reference_img : str | nib.Nifti1Image | None
            NIfTI reference for ROI resampling; required if roi_name is provided.

        Returns
        -------
        (vmin, vmax) : tuple[float, float]
            Global min/max over all subjects (optionally within ROI).
        """
        # Lazy import so this function can run even without nibabel when ROI is unused
        if roi_name is not None:
            try:
                import nibabel as nib  # noqa: F401
            except Exception as e:
                raise ImportError("ROI masking requires nibabel installed.") from e
            if reference_img is None:
                raise ValueError("`reference_img` is required when `roi_name` is provided.")

        gmin = np.inf
        gmax = -np.inf
        any_val = False

        for sid in X_train.index:
            # ---- get subject array (your provided helper) ----
            arr = BrainVisualizer.get_jmap_array(sid, X_train)

            # Ensure a 3D volume if a 4D array shows up (take first timepoint)
            if isinstance(arr, np.ndarray) and arr.ndim == 4:
                arr = arr[..., 0]

            # ---- ROI masking (added block) ----
            if roi_name is not None:
                if not (isinstance(arr, np.ndarray) and arr.ndim == 3):
                    raise ValueError("ROI selection requires a 3D volume per subject.")
                ref_img = reference_img
                if isinstance(reference_img, str):
                    import nibabel as nib
                    ref_img = nib.load(reference_img)
                # use your class helper to mask to the requested ROI
                arr = self._mask_with_roi(arr, ref_img, roi_name)

            # ---- Reduction: fast path for numeric ndarrays; fallback streaming otherwise ----
            if isinstance(arr, np.ndarray) and np.issubdtype(arr.dtype, np.number):
                if arr.size:
                    mn = np.nanmin(arr)
                    mx = np.nanmax(arr)
                    if np.isfinite(mn) and np.isfinite(mx):
                        any_val = True
                        if mn < gmin: gmin = mn
                        if mx > gmax: gmax = mx
            else:
                # object/ragged/string cases
                for v in BrainVisualizer._stream_numeric(arr):
                    any_val = True
                    if v < gmin: gmin = v
                    if v > gmax: gmax = v

        if not any_val:
            raise ValueError("No finite numeric voxel values found (after ROI masking, if applied).")

        return (float(gmin), float(gmax))


    def plot_brain_roi_3d_vedo(
        self,
        roi_name: str,
        out_path: str = "brain_roi.png",
        atlas: str = "cort-maxprob-thr25-2mm",   # or "sub-maxprob-thr25-2mm"
        brain_alpha: float = 0.05,
        roi_alpha: float = 0.85,
    ):
        """
        Render a transparent brain outline (MNI152 mask) with the selected
        Harvard–Oxford ROI filled in blue. Saves PNG to `out_path`.
        Compatible with different nilearn and vedo versions.
        """
        import numpy as np
        import nibabel as nib
        from nilearn import datasets, image
        from vedo import settings, Volume, show

        # ---------- Fetch template & atlas ----------
        brain_mask_img = datasets.load_mni152_brain_mask()

        try:
            ho = datasets.fetch_atlas_harvard_oxford(atlas)
        except TypeError:
            ho = datasets.fetch_atlas_harvard_oxford(atlas_name=atlas)

        ho_maps = ho.get("maps", getattr(ho, "maps", None))
        if isinstance(ho_maps, nib.spatialimages.SpatialImage):
            atlas_img = ho_maps
        else:
            atlas_img = nib.load(ho_maps)

        labels = ho.get("labels", getattr(ho, "labels", []))
        labels = [lbl.decode("utf-8") if isinstance(lbl, (bytes, bytearray)) else str(lbl) for lbl in labels]

        atlas_img = image.resample_to_img(
            atlas_img, brain_mask_img,
            interpolation="nearest",
            force_resample=True,
            copy_header=True,
        )

        # ---------- ROI mask ----------
        target = roi_name.casefold().strip()
        idx = [i for i, name in enumerate(labels) if target in name.casefold()]
        if not idx:
            sample = ", ".join([n for n in labels[1:10] if n])
            raise ValueError(f"ROI '{roi_name}' not found. Example labels: {sample}")

        atlas_data = atlas_img.get_fdata().astype(int)
        roi_mask = np.isin(atlas_data, idx).astype(np.uint8)

        brain_data = (brain_mask_img.get_fdata() > 0).astype(np.uint8)

        from nibabel.affines import voxel_sizes
        spacing = tuple(float(s) for s in voxel_sizes(brain_mask_img.affine))

        # ---------- vedo meshes ----------
        brain_vol = Volume(brain_data, spacing=spacing)
        brain_mesh = brain_vol.isosurface(0.5).alpha(brain_alpha).c("gray5")

        # explicit direction avoids camera requirement
        edge = brain_mesh.silhouette(direction=(0, 0, 1)).c("black").linewidth(2).alpha(0.9)

        roi_vol = Volume(roi_mask, spacing=spacing)
        roi_mesh = roi_vol.isosurface(0.5).alpha(roi_alpha).c("royalblue")

        # ---------- Render off-screen ----------
        settings.default_backend = "vtk"

        # enable hidden line removal if your vedo has the setting
        if hasattr(settings, "useHiddenLineRemoval"):
            settings.useHiddenLineRemoval = True

        # Create the plot without specifying camera dict
        plt = show(
            [brain_mesh, edge, roi_mesh],
            bg="white",
            axes=0,
            size=(1200, 900),
            interactive=False,
            offscreen=True,
        )

        # Now adjust camera orientation similar to:
        # p.camera.azimuth = -60
        # p.camera.elevation = 20
        # p.camera.zoom(1.2)
        plt.camera.Azimuth(-60)
        plt.camera.Elevation(20)
        # extra counter-clockwise 90° spin around Z
        
        plt.camera.Zoom(1.2)
        plt.render()

        # view 1
        # # p.camera.azimuth = -60
        # # p.camera.elevation = 20
        # # p.camera.zoom(1.2)
        # plt.camera.Azimuth(0)
        # plt.camera.Elevation(0)
        # # extra counter-clockwise 90° spin around Z
        
        # plt.camera.Zoom(1.2)

        # view 2
        # plt.camera.Azimuth(-60)
        # plt.camera.Elevation(20)
        # # extra counter-clockwise 90° spin around Z
        
        # #plt.camera.Azimuth(-90)
        # plt.camera.Roll(90)
        # plt.camera.Elevation(0)
        # plt.camera.Zoom(1.2)

        # view 3
        # plt.camera.Azimuth(-60)
        # plt.camera.Elevation(20)
        # # extra counter-clockwise 90° spin around Z
        
        # #plt.camera.Azimuth(-90)
        # plt.camera.Roll(-90)
        # plt.camera.Elevation(270)
        # plt.camera.Zoom(1.2)

        plt.screenshot(out_path)
        plt.close()
        return out_path



    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _print_available_rois(self) -> None:
        """Fetch and print available Harvard–Oxford ROI labels."""
        if datasets is None:
            raise ImportError("nilearn is required to list ROIs.")
        atlas = datasets.fetch_atlas_harvard_oxford(self.atlas_name)
        roi_list = [str(l) for l in atlas.labels if l not in (0, "Background")]
        print("Available ROIs:")
        for r in roi_list:
            print(" -", r)

    def _mask_with_roi(
        self,
        vol: np.ndarray,
        ref_img: "nib.Nifti1Image",
        roi_name: str,
    ) -> np.ndarray:
        """Apply Harvard–Oxford ROI mask to a 3D volume.

        Notes
        -----
        - Resamples the atlas onto ``ref_img`` grid and, if needed, onto the
          current ``vol`` grid. Uses nearest-neighbor to preserve labels.
        - Matches ROI names case-insensitively; exact match preferred, otherwise
          a unique substring match is accepted; otherwise an error is raised.
        """
        assert vol.ndim == 3, "ROI masking requires a 3D volume"
        if nib is None or datasets is None or image is None:
            raise ImportError("nibabel and nilearn must be installed to use ROI masking.")

        atlas = datasets.fetch_atlas_harvard_oxford(self.atlas_name)
        atlas_img = image.resample_to_img(
            atlas.maps, ref_img, interpolation="nearest", force_resample=True, copy_header=True
        )
        atlas_arr = np.asarray(image.get_data(atlas_img))  # same grid as ref_img
        atlas_labels = [str(l) for l in atlas.labels]

        # ROI matching
        q = roi_name.strip().lower()
        exact_matches = [i for i, n in enumerate(atlas_labels) if n.lower() == q]
        if len(exact_matches) == 1:
            roi_idx = exact_matches[0]
        else:
            subs = [i for i, n in enumerate(atlas_labels) if q in n.lower()]
            subs = [i for i in subs if i != 0]  # skip background
            if len(subs) == 1:
                roi_idx = subs[0]
            elif len(subs) > 1:
                cand = [atlas_labels[i] for i in subs]
                raise ValueError(
                    f"ROI name {roi_name!r} is ambiguous. Candidates: {cand}"
                )
            else:
                raise ValueError(f"ROI name {roi_name!r} not found in Harvard–Oxford atlas.")

        if roi_idx == 0:
            raise ValueError("Selected ROI is background; choose a labeled region.")

        # If ref_img grid doesn't match vol shape (e.g., you downsampled), resample atlas to vol
        if atlas_arr.shape != vol.shape:
            vol_img_like = nib.Nifti1Image(vol.astype(np.float32), ref_img.affine, ref_img.header)
            atlas_img = image.resample_to_img(
                atlas_img, vol_img_like, interpolation="nearest", force_resample=True, copy_header=True
            )
            atlas_arr = np.asarray(image.get_data(atlas_img))
            if atlas_arr.shape != vol.shape:
                raise ValueError(
                    f"Atlas grid {atlas_arr.shape} does not match image grid {vol.shape} after resampling."
                )

        roi_mask = atlas_arr == roi_idx
        if not np.any(roi_mask):
            label = atlas_labels[roi_idx] if 0 <= roi_idx < len(atlas_labels) else str(roi_idx)
            raise ValueError(f"No voxels found for ROI '{label}' on the target grid.")

        return np.where(roi_mask, vol, np.nan)

if __name__ == "__main__":
    from nilearn import datasets
    ref_img = datasets.load_mni152_template(resolution=2)
    roi_name = "Temporal Fusiform Cortex, posterior division"
    output_dir = '/home/junfu.cheng/SMILE/github/j_map_2025_8_16/figure_brain/figure/jmap_visulize/whole_brain'
    filename = "temporal_fusiform_post.png"
    png_path = os.path.join(output_dir, filename)
    viz = BrainVisualizer()
    png_path = viz.plot_brain_roi_3d_vedo(
        roi_name="Temporal Fusiform Cortex, posterior division",
        out_path=png_path
    )
    print("Saved to:", png_path)
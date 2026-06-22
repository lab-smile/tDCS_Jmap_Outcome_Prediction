"""
brainviz_pro.py

Upgraded utilities to work with jMAP arrays and visualize brain images, with
3D ROI rendering improvements.

What's new in this revision
---------------------------
1) Camera presets via Enum (CameraView) that the user can select for 3D ROI plots.
2) Output directory & filename arguments for 3D plots; always saves PNG, SVG, and PDF.
3) Multi-ROI support: the user can pass a mapping of ROI -> {color, alpha} to render
   several ROIs in one scene, each with its own style.
4) Before plotting, the function prints all available Harvard–Oxford ROI names and
   a list of color name options the user may pick from.
5) Minor fixes: missing imports, safer saves, helper to convert PNG -> SVG/PDF.
6) **NEW:** Optional **subcortical structure rendering** from the Harvard–Oxford
   subcortical atlas (default: "sub-maxprob-thr25-2mm"). You can pass
   `subcortical_rois` and `subcortical_styles` just like cortical ROIs, and the
   function will render them into the same 3D scene. The function also prints the
   available subcortical structure names.

Note
----
SVG/PDF are created by re-embedding the PNG screenshot into vector containers via
matplotlib. The image content remains raster (as VTK/vedo screenshots are raster).
"""
from __future__ import annotations

import os
import ast
import numbers
import datetime as _dt
from typing import Dict, Optional, Union, Any, Tuple, Iterable, List, Mapping
from enum import Enum

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy.ndimage import zoom
from scipy.ndimage import gaussian_filter, binary_erosion

from nibabel import Nifti1Image, save as save_nii  # keep this
# Optional neuroimaging stack
try:
    import nibabel as nib  # type: ignore
    from nilearn import datasets, image  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    nib = None  # type: ignore
    datasets = None  # type: ignore
    image = None  # type: ignore


__all__ = ["BrainVisualizer", "CameraView"]


class CameraView(Enum):
    """Presets for 3D camera orientation (vedo/VTK)."""
    ISO = "iso"         # az=-60, el=20, zoom=1.2 (default)
    ORTHO = "ortho"     # az=0, el=0, zoom=1.2 (like view 1 in notes)
    ROLL90 = "roll90"   # roll +90 then elevate, zoom (like view 2 idea)
    ROLL270 = "roll270" # roll -90 then elevate 270, zoom (like view 3 idea)


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
                pass
        return np.array(value)

    # ------------------------------------------------------------------
    # 3D multi-slice layouts (grid, cloud, histogram) — unchanged
    #  (snipped in this excerpt to keep focus on ROI upgrades)
    # ------------------------------------------------------------------

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
        """Visualize a brain image (2D/3D/4D) and optionally save it."""
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
                raise ImportError("nibabel and nilearn must be installed to use ROI masking.")

            ref_img = nib.load(reference_img) if isinstance(reference_img, str) else reference_img
            vol = self._mask_with_roi(vol, ref_img, roi_name)
            title_suffix += f"  [ROI: {roi_name}]"

        # ---- Colormap + smoothing ----
        interp_method = "bilinear" if smooth else "nearest"
        cm = plt.get_cmap(cmap)
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
        plt.close(fig)
        return saved_paths

    @staticmethod
    def save_as_nii(vol, reference_img, out_dir, base, saved_paths):
        # >>> Save as NIfTI (.nii.gz) if 3D
        if vol.ndim == 3:
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
            print('[BrainVisualizer] vol.ndim != 3, skipping NIfTI save')
        return saved_paths

    # ------------------------------------------------------------------
    # NEW: 3D Harvard–Oxford ROI rendering with multiple ROIs and camera presets
    #       **and optional subcortical structure rendering**
    # ------------------------------------------------------------------
    def plot_brain_rois_3d_vedo(
        self,
        rois: Optional[Iterable[str]] = None,
        roi_styles: Optional[Mapping[str, Mapping[str, Any]]] = None,
        out_dir: str = "jmap_visulize/rois",
        filename: str = "brain_rois",
        atlas: str = "cort-maxprob-thr25-2mm",
        brain_alpha: float = 0.05,
        view: CameraView = CameraView.ISO,
        list_only: bool = False,
        *,
        # --- NEW subcortical options ---
        subcortical_rois: Optional[Iterable[str]] = None,
        subcortical_styles: Optional[Mapping[str, Mapping[str, Any]]] = None,
        subcortical_atlas: str = "sub-maxprob-thr25-2mm",
    ) -> Optional[Dict[str, str]]:
        """
        Render a transparent brain outline (MNI152 mask) with one or more
        Harvard–Oxford ROIs **and optional subcortical structures**, each with
        configurable color and alpha.

        Parameters
        ----------
        rois : iterable of str, optional
            Cortical ROI names (case-insensitive; substring match allowed if unique).
        roi_styles : mapping str -> {"color": <name>, "alpha": float}, optional
            Per-ROI style overrides for cortical ROIs.
        out_dir : str
            Output directory for figures. Will be created if it does not exist.
        filename : str
            Base filename (no extension). Files saved as PNG/SVG/PDF.
        atlas : str
            Harvard–Oxford **cortical** atlas variant, e.g., 'cort-maxprob-thr25-2mm'.
        brain_alpha : float
            Transparency of the whole-brain outline.
        view : CameraView
            Camera preset orientation.
        list_only : bool
            If True, only prints options (ROIs and color names) and returns.
        subcortical_rois : iterable of str, optional
            Names from the Harvard–Oxford **subcortical** atlas. Same matching rules as `rois`.
        subcortical_styles : mapping str -> {"color": <name>, "alpha": float}, optional
            Per-ROI style overrides for subcortical structures.
        subcortical_atlas : str
            Harvard–Oxford subcortical atlas ID (default 'sub-maxprob-thr25-2mm').

        Returns
        -------
        dict | None
            Mapping of saved file paths (png, svg, pdf) if figures were saved.
        """
        # ---------- Print ROI and color options up front ----------
        cortical_labels = self._get_available_rois(atlas)
        print("Available **cortical** ROIs (Harvard–Oxford, {}):".format(atlas))
        for r in cortical_labels:
            print(" -", r)

        if datasets is None:
            raise ImportError("nilearn is required to list and render ROIs.")

        # Subcortical list
        try:
            sub_labels = self._get_available_rois(subcortical_atlas)
        except Exception:
            sub_labels = []
        if sub_labels:
            print("\nAvailable **subcortical** structures (Harvard–Oxford, {}):".format(subcortical_atlas))
            for r in sub_labels:
                print(" -", r)
        else:
            print("\n(No subcortical atlas labels could be fetched; check 'subcortical_atlas' and nilearn version.)")

        named_colors = sorted(matplotlib.colors.get_named_colors_mapping().keys())
        common_colors = [
            "red", "green", "blue", "royalblue", "dodgerblue", "navy",
            "orange", "gold", "yellow", "purple", "violet", "magenta",
            "pink", "brown", "black", "white", "gray", "slategray", "cyan", "teal"
        ]
        print("\nCommon color options:")
        print(", ".join(common_colors))
        print("\n(Full list has {} names; any matplotlib color name is accepted.)".format(len(named_colors)))

        if list_only:
            return None

        # ---------- Import vedo only when needed ----------
        from vedo import settings, Volume, show

        # ---------- Fetch template & atlas ----------
        brain_mask_img = datasets.load_mni152_brain_mask()

        # Cortical atlas fetch (backward-compat keyword)
        try:
            ho_cort = datasets.fetch_atlas_harvard_oxford(atlas)
        except TypeError:
            ho_cort = datasets.fetch_atlas_harvard_oxford(atlas_name=atlas)

        cort_maps = ho_cort.get("maps", getattr(ho_cort, "maps", None))
        if isinstance(cort_maps, nib.spatialimages.SpatialImage):
            cort_img = cort_maps
        else:
            cort_img = nib.load(cort_maps)

        # Subcortical atlas (optional)
        ho_sub = None
        sub_img = None
        if subcortical_rois:
            try:
                ho_sub = datasets.fetch_atlas_harvard_oxford(subcortical_atlas)
            except TypeError:
                ho_sub = datasets.fetch_atlas_harvard_oxford(atlas_name=subcortical_atlas)
            sub_maps = ho_sub.get("maps", getattr(ho_sub, "maps", None))
            sub_img = sub_maps if isinstance(sub_maps, nib.spatialimages.SpatialImage) else nib.load(sub_maps)

        # Resample atlases to MNI brain mask grid
        cort_img = image.resample_to_img(
            cort_img, brain_mask_img, interpolation="nearest", force_resample=True, copy_header=True
        )
        if sub_img is not None:
            sub_img = image.resample_to_img(
                sub_img, brain_mask_img, interpolation="nearest", force_resample=True, copy_header=True
            )

        # ---------- Prepare brain & spacing ----------
        brain_data = (brain_mask_img.get_fdata() > 0).astype(np.uint8)
        from nibabel.affines import voxel_sizes
        spacing = tuple(float(s) for s in voxel_sizes(brain_mask_img.affine))

        # ---------- vedo settings ----------
        settings.default_backend = "vtk"
        if hasattr(settings, "useHiddenLineRemoval"):
            settings.useHiddenLineRemoval = True

        # ---------- Build meshes ----------
        brain_vol = Volume(brain_data, spacing=spacing)
        brain_mesh = brain_vol.isosurface(0.5).alpha(brain_alpha).c("gray5")
        edge = brain_mesh.silhouette(direction=(0, 0, 1)).c("black").linewidth(2).alpha(0.9)

        actors = [brain_mesh, edge]

        # ---------- Internal helpers ----------
        def _labels_list(atlas_obj) -> List[str]:
            labels = atlas_obj.get("labels", getattr(atlas_obj, "labels", []))
            return [l.decode("utf-8") if isinstance(l, (bytes, bytearray)) else str(l) for l in labels]

        def _atlas_data(img_like) -> np.ndarray:
            return img_like.get_fdata().astype(int)

        def _find_indices(name: str, labels_cf: List[str]) -> List[int]:
            q = name.casefold().strip()
            exact = [i for i, n in enumerate(labels_cf) if i != 0 and n == q]
            if len(exact) == 1:
                return exact
            subs = [i for i, n in enumerate(labels_cf) if i != 0 and q in n]
            subs = list(dict.fromkeys(subs))
            if not subs:
                raise ValueError(f"ROI '{name}' not found in atlas labels.")
            return subs

        # ---------- Resolve & add cortical meshes ----------
        if rois:
            cort_labels = _labels_list(ho_cort)
            cort_cf = [str(l).casefold() for l in cort_labels]
            cort_arr = _atlas_data(cort_img)
            for roi in rois:
                idxs = _find_indices(roi, cort_cf)
                roi_mask = np.isin(cort_arr, idxs).astype(np.uint8)
                if roi_mask.sum() == 0:
                    continue
                v = Volume(roi_mask, spacing=spacing)
                mesh = v.isosurface(0.5)
                color = "royalblue"
                alpha = 0.85
                if roi_styles and roi in roi_styles:
                    sty = roi_styles[roi]
                    if "color" in sty:
                        color = sty["color"]
                    if "alpha" in sty:
                        alpha = float(sty["alpha"])
                mesh = mesh.c(color).alpha(alpha)
                actors.append(mesh)

        # ---------- Resolve & add subcortical meshes (NEW) ----------
        if subcortical_rois and ho_sub is not None and sub_img is not None:
            sub_labels = _labels_list(ho_sub)
            sub_cf = [str(l).casefold() for l in sub_labels]
            sub_arr = _atlas_data(sub_img)
            for roi in subcortical_rois:
                idxs = _find_indices(roi, sub_cf)
                roi_mask = np.isin(sub_arr, idxs).astype(np.uint8)
                if roi_mask.sum() == 0:
                    continue
                v = Volume(roi_mask, spacing=spacing)
                mesh = v.isosurface(0.5)
                color = "seagreen"
                alpha = 0.85
                if subcortical_styles and roi in subcortical_styles:
                    sty = subcortical_styles[roi]
                    if "color" in sty:
                        color = sty["color"]
                    if "alpha" in sty:
                        alpha = float(sty["alpha"])
                mesh = mesh.c(color).alpha(alpha)
                actors.append(mesh)

        # ---------- Render off-screen ----------
        plt3d = show(
            actors,
            bg="white",
            axes=0,
            size=(1600, 1200),
            interactive=False,
            offscreen=True,
        )

        # ---------- Camera presets ----------
        def apply_camera(v: CameraView):
            if v == CameraView.ISO:
                plt3d.camera.Azimuth(-60)
                plt3d.camera.Elevation(20)
                plt3d.camera.Zoom(1.2)
            elif v == CameraView.ORTHO:
                plt3d.camera.Azimuth(0)
                plt3d.camera.Elevation(0)
                plt3d.camera.Zoom(1.2)
            elif v == CameraView.ROLL90:
                plt3d.camera.Azimuth(-60)
                plt3d.camera.Elevation(20)
                plt3d.camera.Roll(90)
                plt3d.camera.Elevation(0)
                plt3d.camera.Zoom(1.2)
            elif v == CameraView.ROLL270:
                plt3d.camera.Azimuth(-60)
                plt3d.camera.Elevation(20)
                plt3d.camera.Roll(-90)
                plt3d.camera.Elevation(270)
                plt3d.camera.Zoom(1.2)
            else:
                plt3d.camera.Azimuth(-60)
                plt3d.camera.Elevation(20)
                plt3d.camera.Zoom(1.2)

        apply_camera(view)
        plt3d.render()

        # ---------- Save outputs ----------
        os.makedirs(out_dir, exist_ok=True)
        timestamp = _dt.datetime.now().strftime('%Y%m%d-%H%M%S')
        base = os.path.splitext(filename)[0]
        stem = os.path.join(out_dir, f"{base}_{timestamp}")

        png_path = stem + ".png"
        plt3d.screenshot(png_path)
        plt3d.close()

        # Convert PNG -> PDF/SVG via matplotlib
        svg_path = stem + ".svg"
        pdf_path = stem + ".pdf"
        self._png_to_svg_pdf(png_path, svg_path, pdf_path)

        print("Saved figure(s):")
        print(" -", png_path)
        print(" -", svg_path)
        print(" -", pdf_path)
        return {"png": png_path, "svg": svg_path, "pdf": pdf_path}

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _print_available_rois(self) -> None:
        """Fetch and print available Harvard–Oxford ROI labels (cortical only, using self.atlas_name)."""
        roi_list = self._get_available_rois(self.atlas_name)
        print("Available ROIs:")
        for r in roi_list:
            print(" -", r)

    def _get_available_rois(self, atlas_name: str) -> List[str]:
        if datasets is None:
            raise ImportError("nilearn is required to list ROIs.")
        atlas = datasets.fetch_atlas_harvard_oxford(atlas_name)
        labels = atlas.get("labels", getattr(atlas, "labels", []))
        roi_list = [str(l) for l in labels if l not in (0, "Background")]
        return roi_list

    def _mask_with_roi(
        self,
        vol: np.ndarray,
        ref_img: "nib.Nifti1Image",
        roi_name: str,
    ) -> np.ndarray:
        """Apply Harvard–Oxford ROI mask to a 3D volume."""
        assert vol.ndim == 3, "ROI masking requires a 3D volume"
        if nib is None or datasets is None or image is None:
            raise ImportError("nibabel and nilearn must be installed to use ROI masking.")

        atlas = datasets.fetch_atlas_harvard_oxford(self.atlas_name)
        atlas_img = image.resample_to_img(
            atlas.maps, ref_img, interpolation="nearest", force_resample=True, copy_header=True
        )
        # deprecated nilearn.get_data; use get_fdata
        atlas_arr = np.asarray(atlas_img.get_fdata())
        atlas_labels = [str(l) for l in atlas.labels]

        # ROI matching
        q = roi_name.strip().lower()
        exact_matches = [i for i, n in enumerate(atlas_labels) if str(n).lower() == q]
        if len(exact_matches) == 1:
            roi_idx = exact_matches[0]
        else:
            subs = [i for i, n in enumerate(atlas_labels) if q in str(n).lower()]
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

        # If ref_img grid doesn't match vol shape, resample atlas to vol
        if atlas_arr.shape != vol.shape:
            vol_img_like = nib.Nifti1Image(vol.astype(np.float32), ref_img.affine, ref_img.header)
            atlas_img = image.resample_to_img(
                atlas_img, vol_img_like, interpolation="nearest", force_resample=True, copy_header=True
            )
            atlas_arr = np.asarray(atlas_img.get_fdata())
            if atlas_arr.shape != vol.shape:
                raise ValueError(
                    f"Atlas grid {atlas_arr.shape} does not match image grid {vol.shape} after resampling."
                )

        roi_mask = atlas_arr == roi_idx
        if not np.any(roi_mask):
            label = atlas_labels[roi_idx] if 0 <= roi_idx < len(atlas_labels) else str(roi_idx)
            raise ValueError(f"No voxels found for ROI '{label}' on the target grid.")

        return np.where(roi_mask, vol, np.nan)

    @staticmethod
    def _png_to_svg_pdf(png_path: str, svg_path: str, pdf_path: str) -> None:
        """Wrap a PNG into SVG and PDF via matplotlib without external deps."""
        import matplotlib.pyplot as plt
        import matplotlib.image as mpimg

        img = mpimg.imread(png_path)

        # Save as SVG
        fig_svg = plt.figure(figsize=(img.shape[1] / 100, img.shape[0] / 100), dpi=100)
        ax = fig_svg.add_subplot(111)
        ax.imshow(img)
        ax.axis('off')
        fig_svg.savefig(svg_path, format='svg', bbox_inches='tight', pad_inches=0)
        plt.close(fig_svg)

        # Save as PDF
        fig_pdf = plt.figure(figsize=(img.shape[1] / 100, img.shape[0] / 100), dpi=100)
        ax = fig_pdf.add_subplot(111)
        ax.imshow(img)
        ax.axis('off')
        fig_pdf.savefig(pdf_path, format='pdf', bbox_inches='tight', pad_inches=0)
        plt.close(fig_pdf)


if __name__ == "__main__":
    # Example usage including subcortical additions
    viz = BrainVisualizer()
    viz.plot_brain_rois_3d_vedo(
        rois = [
            "Temporal Fusiform Cortex, posterior division",
            "Frontal Medial Cortex",
            "Frontal Orbital Cortex",
            "Subcallosal Cortex",
        ],
        roi_styles = {
            "Temporal Fusiform Cortex, posterior division": {"color": "royalblue", "alpha": 0.9},
            "Frontal Medial Cortex": {"color": "red", "alpha": 0.9},
            "Frontal Orbital Cortex": {"color": "red", "alpha": 0.9},
            "Subcallosal Cortex": {"color": "red", "alpha": 0.9},
        },
        # NEW: add subcortical structures
        subcortical_rois=[
            "Left Amygdala",
            "Right Amygdala",
        ],
        subcortical_styles={
            "Left Amygdala": {"color": "gold", "alpha": 1},
            "Right Amygdala": {"color": "gold", "alpha": 1},
        },
        out_dir='/home/junfu.cheng/SMILE/github/j_map_fake/figure_brain/figure/whole_brain',
        filename="fusiform_pfc_subcortical_scene",
        view=CameraView.ROLL90,
        brain_alpha=0.06,
        list_only=False
    )

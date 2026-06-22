import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    f1_score,
    matthews_corrcoef,
    recall_score,
    confusion_matrix,
    balanced_accuracy_score,
    classification_report
)

# histogram plotting
import matplotlib.pyplot as plt
from scipy import stats
import shap
import os

class ModelEvaluator:
    def __init__(self, verbose: bool = False, printer = print):
        self.verbose = verbose
        self.printer = printer

    def evaluate(self, model_pipeline, X_test, y_test):
        y_pred = model_pipeline.predict(X_test)

        # If you have probabilities (e.g., for AUC)
        try:
            y_proba = model_pipeline.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_proba)
        except:
            auc = None  # Some models (e.g., SVM without probability=True) won't support this

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='binary')  # change average for multiclass
        f1_responder = f1_score(y_test, y_pred, pos_label=1)  # Responder (class 1)
        f1_non_responder = f1_score(y_test, y_pred, pos_label=0)  # Non-responder (class 0)
        mcc = matthews_corrcoef(y_test, y_pred)
        recall = recall_score(y_test, y_pred, average='binary')
        bal_acc = balanced_accuracy_score(y_test, y_pred)

        # Specificity = TN / (TN + FP)
        if len(np.unique(y_test)) < 2 or len(np.unique(y_test))>2:
            raise ValueError("[Warning] Only one class present in y_test. Metrics may be unreliable.")
        labels = [0, 1]
        cm = confusion_matrix(y_test, y_pred, labels=labels)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        else:
            specificity = None  # Not defined for multiclass

        report = classification_report(y_test, y_pred, output_dict=True, labels=labels)

        if self.verbose:
            self.printer(f"[Evaluator] Accuracy: {acc:.4f}\n")
            self.printer(f"[Evaluator] AUC: {auc:.4f}\n" if auc is not None else "[Evaluator] AUC: N/A\n")
            self.printer(f"[Evaluator] F1 Score: {f1:.4f}\n")
            self.printer(f"F1 Score (Responder - Class 1): {f1_responder}\n")
            self.printer(f"F1 Score (Non-Responder - Class 0): {f1_non_responder}\n")
            self.printer(f"[Evaluator] MCC: {mcc:.4f}\n")
            self.printer(f"[Evaluator] Sensitivity (Recall): {recall:.4f}\n")
            self.printer(f"[Evaluator] Specificity: {specificity:.4f}\n" if specificity is not None else "[Evaluator] Specificity: N/A")
            self.printer(f"[Evaluator] Balanced Accuracy: {bal_acc:.4f}\n")

        return {
            "accuracy": acc,
            "auc": auc,
            "f1": f1,
            "mcc": mcc,
            "sensitivity": recall,
            "specificity": specificity,
            "balanced_accuracy": bal_acc,
            "report": report
        }

    def get_coordinate_table_mni_shap(self,
                                    pipeline,
                                    X_test,
                                    y_test=None,
                                    topk_voxels=500,
                                    class_index=1,
                                    background_size=50,
                                    sample_size=100,
                                    use_signed=True,
                                    verbose=True):
        """
        Explain a complex pipeline:
            prep(flatten) -> pca -> ttest -> mrmr -> rbf -> sgd
        using SHAP on the *selected PCA features*, then back-project SHAP
        to voxel space and return a coordinate table (MNI, region, SHAP).

        Parameters
        ----------
        pipeline : sklearn.Pipeline (fitted)
        X_test   : pd.DataFrame   (contains the J-map columns used by `prep`)
        y_test   : array-like or None (unused)
        topk_voxels : int         Number of voxels to return after ranking by |SHAP|
        class_index : int         Positive class index for predict_proba
        background_size : int     SHAP background size (Permutation/Independent masker)
        sample_size    : int      Number of rows to explain
        use_signed     : bool     If True, back-project mean signed SHAP; else use abs
        verbose        : bool

        Returns
        -------
        pd.DataFrame with columns:
            ['rank', 'mean_shap_voxel', 'mean_abs_shap_voxel',
            'voxel_ijk', 'channel', 'MNI_xyz', 'region']
        """
        import numpy as np
        import pandas as pd
        import shap
        from sklearn.pipeline import Pipeline

        # --------- sanity checks & step extraction ----------
        req = ["prep", "pca", "ttest", "mrmr"]
        for k in req:
            if k not in pipeline.named_steps:
                raise ValueError(f"Pipeline must include a '{k}' step.")
        prep  = pipeline.named_steps["prep"]
        pca   = pipeline.named_steps["pca"]
        ttest = pipeline.named_steps["ttest"]
        mrmr  = pipeline.named_steps["mrmr"]

        # Build tail after mrmr (e.g., rbf -> sgd)
        after_keys = []
        seen = False
        for k, _ in pipeline.steps:
            if k == "mrmr":
                seen = True
                continue
            if seen:
                after_keys.append(k)
        rest_after = Pipeline([(k, pipeline.named_steps[k]) for k in after_keys])
        if verbose:
            head = after_keys[0] if after_keys else "END"
            print(f"[INFO] Explaining on features after 'mrmr' and before {head}; tail steps = {after_keys}")

        # --------- forward to selected PCA space ----------
        # 1) flatten -> X_flat (DataFrame with jmap_flat_* columns)
        X_flat = prep.transform(X_test)

        # 2) PCA -> X_pca (ensure DataFrame with consistent names)
        Z = pca.transform(X_flat)
        # Determine PCA output names and components no matter vanilla PCA or PCAWithNames
        if hasattr(pca, "get_feature_names_out"):
            pca_names = np.array(pca.get_feature_names_out()).astype(str)
        else:
            n_out = getattr(pca, "n_components_", None)
            if n_out is None and hasattr(pca, "components_"):
                n_out = pca.components_.shape[0]
            if n_out is None and hasattr(pca, "pca_") and hasattr(pca.pca_, "components_"):
                n_out = pca.pca_.components_.shape[0]
            if n_out is None:
                n_out = Z.shape[1]
            pca_names = np.array([f"f{i}" for i in range(n_out)], dtype=str)

        X_pca = pd.DataFrame(Z, index=X_flat.index, columns=pca_names)

        # 3) t-test selector (expects DataFrame with training-time column names)
        X_t = ttest.transform(X_pca)
        if not isinstance(X_t, pd.DataFrame):
            # try to wrap with names the selector reports
            if hasattr(ttest, "get_feature_names_out"):
                cols_t = ttest.get_feature_names_out()
            else:
                cols_t = getattr(ttest, "selected_columns_", None)
            if cols_t is None:
                raise TypeError("WelchTTestSelector returned ndarray and no names are available.")
            X_t = pd.DataFrame(X_t, index=X_pca.index, columns=cols_t)

        # 4) mRMR selector
        X_sel = mrmr.transform(X_t)
        if not isinstance(X_sel, pd.DataFrame):
            if hasattr(mrmr, "get_feature_names_out"):
                cols_m = mrmr.get_feature_names_out()
            else:
                cols_m = getattr(mrmr, "selected_columns_", None)
            if cols_m is None:
                raise TypeError("MRMRSelector returned ndarray and no names are available.")
            X_sel = pd.DataFrame(X_sel, index=X_t.index, columns=cols_m)

        sel_names = np.array(X_sel.columns).astype(str)
        if sel_names.size == 0:
            raise ValueError("No features selected by MRMRSelector; adjust thresholds or K.")

        # Map selected feature names -> PCA component indices robustly
        name_to_idx = {name: i for i, name in enumerate(pca_names)}
        try:
            sel_idx = np.array([name_to_idx[n] for n in sel_names], dtype=int)
        except KeyError as e:
            missing = [n for n in sel_names if n not in name_to_idx]
            raise KeyError(f"Selected feature(s) not found in PCA output names: {missing}") from e

        # --------- SHAP on the small selected-PC space ----------
        # Subsample to keep PermutationExplainer tractable
        n_bg = min(background_size, len(X_sel))
        n_ex = min(sample_size, len(X_sel))
        bg = X_sel.sample(n_bg, random_state=0)
        X_explain = X_sel.iloc[:n_ex]

        # function over selected-PC inputs using the tail (rbf -> sgd)
        if hasattr(rest_after, "predict_proba"):
            def f(Z_in):
                Z_df = pd.DataFrame(Z_in, columns=sel_names)
                P = rest_after.predict_proba(Z_df)
                return P[:, class_index] if P.ndim == 2 else P
        else:
            def f(Z_in):
                Z_df = pd.DataFrame(Z_in, columns=sel_names)
                S = rest_after.decision_function(Z_df)
                return S[:, class_index] if (hasattr(S, "ndim") and S.ndim == 2) else S

        masker = shap.maskers.Independent(bg)
        explainer = shap.Explainer(f, masker)  # chooses PermutationExplainer here
        S = explainer(X_explain)
        V = S.values
        if V.ndim == 3:
            V = V[:, class_index, :]

        # aggregate across samples in selected-PC space
        mean_abs_sel = np.mean(np.abs(V), axis=0)   # shape (m_selected,)
        mean_sel     = np.mean(V, axis=0)

        # --------- back-project SHAP (selected PCs) -> voxel space ----------
        # Get PCA components regardless of wrapper
        if hasattr(pca, "components_"):
            comps = pca.components_                   # (n_pcs, n_voxels)
        elif hasattr(pca, "pca_") and hasattr(pca.pca_, "components_"):
            comps = pca.pca_.components_
        else:
            raise AttributeError("Cannot find PCA components_ on the PCA step.")

        # combine selected components linearly with their (signed) mean SHAP
        vec_sel = mean_sel if use_signed else mean_abs_sel
        voxel_shap_std = (comps[sel_idx, :].T @ vec_sel)   # (n_voxels,)

        # Undo StandardScaler inside `prep` (if present) to express in pre-PCA standardized space
        # Note: per-sample z-scoring inside prep cannot be globally inverted; ranking remains meaningful.
        J = prep
        if getattr(J, "_scaler", None) is not None and hasattr(J._scaler, "scale_"):
            with np.errstate(divide="ignore", invalid="ignore"):
                scale = np.asarray(J._scaler.scale_, dtype=float)
                scale = np.where(scale == 0, 1.0, scale)
                voxel_shap = voxel_shap_std / scale
        else:
            voxel_shap = voxel_shap_std

        # --------- map top voxels -> (i,j,k), MNI, region ----------
        shape = getattr(J, "_example_shape", None)
        if shape is None:
            # ensure example shape is set
            _ = J.transform(X_test.iloc[[0]])
            shape = J._example_shape

        if J.keep_channel_axis and len(shape) == 4:
            xyz_shape, channels = shape[:-1], shape[-1]
        else:
            xyz_shape, channels = shape, 1

        n_vox = voxel_shap.shape[0]
        topk = min(topk_voxels, n_vox)
        order = np.argsort(np.abs(voxel_shap))[::-1][:topk]

        rows = []
        for rank, lin in enumerate(order, start=1):
            voxel_linear = lin // channels
            ch = (lin % channels) if channels > 1 else None
            i, j, k = np.unravel_index(voxel_linear, xyz_shape)
            region = J.voxel_to_region(i, j, k) if (J.atlas_data is not None) else None
            mni    = J.voxel_to_mni(i, j, k)    if (J.atlas_img  is not None) else None
            rows.append({
                "rank": rank,
                "mean_shap_voxel": float(voxel_shap[lin]),
                "mean_abs_shap_voxel": float(abs(voxel_shap[lin])),
                "voxel_ijk": (int(i), int(j), int(k)),
                "channel": ch,
                "MNI_xyz": mni,
                "region": region
            })

        df = pd.DataFrame(rows)
        if verbose:
            print(f"[INFO] Explained {len(X_explain)} samples on {len(sel_names)} selected PCs; "
                f"returned top-{len(df)} voxels.")
        return df




    def cross_validate(self, model_pipeline, X, y, cv_strategy):
        from sklearn.model_selection import cross_val_score
        scores = cross_val_score(model_pipeline, X, y, cv=cv_strategy)
        if self.verbose:
            self.printer(f"[Evaluator] Cross-val scores: {scores}")
        return scores

    

    def vivid_histogram(
        self,
        data,
        test='normality',  # 'normality', 't-test', or None
        population_mean=None,
        alpha=0.05,
        figsize=(10, 6),
        color='skyblue',
        edgecolor='black',
        title="Vivid Histogram",
        experiment_dir="results_unknown_date"
    ):
        data = np.array(data)
        
        if len(data) < 2:
            raise ValueError("Data must contain at least two elements.")

        # === Descriptive stats ===
        mean_val = np.mean(data)
        std_val = np.std(data, ddof=1)
        self.printer(f"Mean: {mean_val:.4f}\n")
        self.printer(f"Standard Deviation: {std_val:.4f}\n")

        # === Compute bin width using Freedman-Diaconis rule ===
        q25, q75 = np.percentile(data, [25, 75])
        iqr = q75 - q25
        bin_width = 2 * iqr * (len(data) ** (-1/3))
        bin_width = bin_width if bin_width > 0 else std_val / 5  # fallback
        bins = int(np.ceil((max(data) - min(data)) / bin_width))
        bins = max(bins, 5)  # ensure a minimum number of bins
        bins = bins * 4

        # === Histogram ===
        plt.figure(figsize=figsize)
        plt.hist(data, bins=bins, color=color, edgecolor=edgecolor, alpha=0.9)
        plt.axvline(mean_val, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_val:.4f}')
        plt.title(title)
        plt.xlabel("Balanced Accuracy")
        plt.ylabel("Frequency")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        plt.tight_layout()
        os.makedirs(os.path.join(experiment_dir, 'figure','balanced_acc'), exist_ok=True)
        plt.savefig(os.path.join(experiment_dir, 'figure','balanced_acc','balanced_acc.png'), dpi=600)
        #plt.show()
        plt.close()

        # === Statistical Tests ===
        if test == 'normality':
            stat, p = stats.normaltest(data)
            self.printer(f"Normality test (D’Agostino and Pearson): stat={stat:.4f}, p={p:.4f}\n")
            if p < alpha:
                self.printer(f"Result: Data does NOT appear normally distributed (p < {alpha})\n")
            else:
                self.printer(f"Result: Data appears normally distributed (p ≥ {alpha})\n")

        elif test == 't-test':
            if population_mean is None:
                raise ValueError("Population mean must be provided for t-test.")
            t_stat, p = stats.ttest_1samp(data, population_mean)
            self.printer(f"One-sample t-test against population mean {population_mean}: t={t_stat:.4f}, p={p:.4f}\n")
            if p < alpha:
                self.printer(f"Result: Mean is significantly different from {population_mean} (p < {alpha})\n")
            else:
                self.printer(f"Result: No significant difference from {population_mean} (p ≥ {alpha})\n")

    def shap_summary_plots(
        self,
        model_pipeline,
        X,
        class_idx: int = 1,
        max_display: int = 20,
        sample_size: int = 1000,
        figsize=(10, 6),
        experiment_dir: str = "results_unknown_date",
        feature_namer=None,              # <-- NEW
    ):
        """
        Compute SHAP values and save summary plots (bar + beeswarm)
        into <experiment_dir>/figure/shap/.
        If feature_namer is provided, column names are prettified for plots/tables.
        """
        import os
        import numpy as np
        import pandas as pd
        import shap
        import matplotlib.pyplot as plt
        from sklearn.pipeline import Pipeline
        from scipy import sparse

        def is_tree_model(est):
            cls = est.__class__.__name__.lower()
            return any(k in cls for k in ["xgb", "lgbm", "gradientboost", "randomforest", "decisiontree", "catboost"])

        def normalize_shap_values(values, n_features, class_idx):
            import numpy as np
            if isinstance(values, list):  # per-class arrays
                arrs = [np.asarray(v) for v in values]
                stacked = np.stack(arrs, axis=1)  # (n, C, F)
                if class_idx < 0 or class_idx >= stacked.shape[1]:
                    raise ValueError(f"class_idx={class_idx} out of bounds for {stacked.shape[1]} classes")
                return stacked[:, class_idx, :], stacked.shape[1]
            vals = np.asarray(values)
            if vals.ndim == 3:
                n, a1, a2 = vals.shape
                if a2 == n_features:          # (n, C, F)
                    C = a1
                    if class_idx < 0 or class_idx >= C:
                        raise ValueError(f"class_idx={class_idx} out of bounds for {C} classes")
                    return vals[:, class_idx, :], C
                if a1 == n_features:          # (n, F, C)  <-- your case
                    C = a2
                    if class_idx < 0 or class_idx >= C:
                        raise ValueError(f"class_idx={class_idx} out of bounds for {C} classes")
                    return vals[:, :, class_idx], C
                raise RuntimeError(f"Unexpected SHAP shape {vals.shape}: neither axis equals n_features={n_features}.")
            if vals.ndim == 2:
                if vals.shape[1] == n_features:
                    return vals, None
                if vals.shape[1] % n_features == 0:
                    C = vals.shape[1] // n_features
                    reshaped = vals.reshape(vals.shape[0], C, n_features)
                    if class_idx < 0 or class_idx >= C:
                        raise ValueError(f"class_idx={class_idx} out of bounds for {C} classes")
                    return reshaped[:, class_idx, :], C
                raise RuntimeError(
                    f"SHAP values have {vals.shape[1]} columns, not equal to n_features={n_features} "
                    f"and not a multiple thereof."
                )
            raise RuntimeError(f"Unsupported SHAP values ndim={vals.ndim}")

        def pick_base_values(base_vals, n_classes_detected, class_idx):
            import numpy as np
            if n_classes_detected in (None, 1):
                return base_vals
            bv = np.asarray(base_vals)
            if bv.ndim == 0:
                return float(bv)
            if bv.ndim == 1 and bv.size == n_classes_detected:
                return bv[class_idx]
            if bv.ndim == 2 and bv.shape[1] == n_classes_detected:
                return bv[:, class_idx]
            return base_vals

        # --- dirs ---
        out_dir = os.path.join(experiment_dir, "figure", "shap")
        os.makedirs(out_dir, exist_ok=True)

        # --- unpack pipeline / estimator ---
        if isinstance(model_pipeline, Pipeline):
            if not model_pipeline.steps:
                raise ValueError("Pipeline has no steps.")
            estimator = model_pipeline.steps[-1][1]
            preprocess = None
            for name, step in model_pipeline.steps[:-1]:
                if name.lower() in ("preprocess", "preprocessing", "preproc"):
                    preprocess = step
                    break
            if preprocess is None:
                for name, step in model_pipeline.steps[:-1]:
                    if hasattr(step, "transform"):
                        preprocess = step
                        break
        else:
            estimator = model_pipeline
            preprocess = None

        # --- transform X (skip SMOTE) ---
        if preprocess is not None and hasattr(preprocess, "transform"):
            X_trans = preprocess.transform(X)
            feature_names = None
            if hasattr(preprocess, "get_feature_names_out"):
                try:
                    feature_names = list(preprocess.get_feature_names_out())
                except Exception:
                    feature_names = None
        else:
            X_trans = X
            feature_names = None

        # --- DataFrame ---
        if isinstance(X_trans, pd.DataFrame):
            X_df = X_trans.copy()
        else:
            n_cols = getattr(X_trans, "shape", (0, 0))[1]
            if feature_names is None and isinstance(X, pd.DataFrame):
                feature_names = [str(c) for c in X.columns]
            if feature_names is not None and len(feature_names) == n_cols:
                X_df = pd.DataFrame(X_trans, columns=feature_names, index=getattr(X, "index", None))
            else:
                X_df = pd.DataFrame(X_trans, columns=[f"f{i}" for i in range(n_cols)], index=getattr(X, "index", None))

        if sparse.issparse(X_df.values):
            X_df = pd.DataFrame(X_df.values.todense(), columns=X_df.columns, index=X_df.index)

        # --- estimator input sanity ---
        n_in = getattr(estimator, "n_features_in_", None)
        if n_in is not None and n_in != X_df.shape[1]:
            raise RuntimeError(
                f"Estimator was trained on {n_in} features but preprocessed X has {X_df.shape[1]}."
            )

        # --- subsample ---
        X_sample = X_df.sample(sample_size, random_state=42) if len(X_df) > sample_size else X_df
        n_feat = X_sample.shape[1]

        # --- compute SHAP ---
        try:
            if is_tree_model(estimator):
                explainer = shap.TreeExplainer(estimator, model_output="raw")
                shap_vals = explainer.shap_values(X_sample.values)
                base_vals = explainer.expected_value
            else:
                if hasattr(estimator, "predict_proba"):
                    f = lambda data: estimator.predict_proba(data)
                elif hasattr(estimator, "decision_function"):
                    f = lambda data: estimator.decision_function(data)
                else:
                    f = lambda data: estimator.predict(data)
                explainer = shap.Explainer(f, X_sample.values, feature_names=X_sample.columns.tolist())
                expl = explainer(X_sample.values)
                shap_vals = expl.values
                base_vals = expl.base_values
        except Exception as e:
            raise RuntimeError(f"Failed to compute SHAP values: {e}")

        # --- normalize to (n, n_features) ---
        values_2d, n_classes_detected = normalize_shap_values(shap_vals, n_feat, class_idx)
        base_vals = pick_base_values(base_vals, n_classes_detected, class_idx)

        if values_2d.shape[1] != n_feat:
            raise RuntimeError(
                f"After normalization, SHAP has {values_2d.shape[1]} features but X has {n_feat}."
            )

        # --- PRETTIFY feature names (NEW) ---
        if feature_namer is not None:
            pretty_cols = [feature_namer(c) for c in X_sample.columns]
        else:
            pretty_cols = list(X_sample.columns)

        # --- Explanation & plots ---
        expl_for_plot = shap.Explanation(
            values=values_2d,
            base_values=base_vals,
            data=X_sample.values,
            feature_names=pretty_cols,   # <-- use pretty names in plots
        )

        # Bar
        plt.figure(figsize=figsize)
        shap.summary_plot(expl_for_plot, plot_type="bar", max_display=max_display, show=False)
        plt.tight_layout()
        bar_path = os.path.join(out_dir, "shap_summary_bar.png")
        plt.savefig(bar_path, dpi=300, bbox_inches="tight")
        plt.close()

        # Beeswarm
        plt.figure(figsize=figsize)
        shap.summary_plot(expl_for_plot, max_display=max_display, show=False)
        plt.tight_layout()
        swarm_path = os.path.join(out_dir, "shap_summary_beeswarm.png")
        plt.savefig(swarm_path, dpi=300, bbox_inches="tight")
        plt.close()

        # Importance table (pretty names)
        mean_abs = np.mean(np.abs(values_2d), axis=0)
        mean_abs_1d = np.asarray(mean_abs).reshape(-1)
        features_1d = pretty_cols

        importance = (
            pd.DataFrame({"feature": features_1d, "mean_abs_shap": mean_abs_1d}, copy=False)
            .sort_values("mean_abs_shap", ascending=False)
            .reset_index(drop=True)
        )

        if getattr(self, "verbose", False):
            self.printer(f"[Evaluator] X_sample shape: {X_sample.shape}\n")
            self.printer(f"[Evaluator] SHAP values (normalized) shape: {values_2d.shape}\n")
            self.printer(f"[Evaluator] SHAP bar saved to: {bar_path}\n")
            self.printer(f"[Evaluator] SHAP beeswarm saved to: {swarm_path}\n")
            self.printer("[Evaluator] Top features by |SHAP|:\n")
            self.printer(importance.head(min(10, len(importance))).to_string(index=False) + "\n")

        return {
            "importance": importance,
            "bar_path": bar_path,
            "beeswarm_path": swarm_path
        }

    def shap_summary_tables(
        self,
        model_pipeline,
        X,
        class_idx: int = 1,
        max_display: int = 20,
        sample_size: int = 1000,
        figsize=(10, 6),
        experiment_dir: str = "results_unknown_date",
        feature_namer=None,              # <-- NEW
    ):
        """
        Compute SHAP values and save summary plots (bar + beeswarm)
        into <experiment_dir>/figure/shap/.
        If feature_namer is provided, column names are prettified for plots/tables.
        """
        import os
        import numpy as np
        import pandas as pd
        import shap
        import matplotlib.pyplot as plt
        from sklearn.pipeline import Pipeline
        from scipy import sparse

        def is_tree_model(est):
            cls = est.__class__.__name__.lower()
            return any(k in cls for k in ["xgb", "lgbm", "gradientboost", "randomforest", "decisiontree", "catboost"])

        def normalize_shap_values(values, n_features, class_idx):
            import numpy as np
            if isinstance(values, list):  # per-class arrays
                arrs = [np.asarray(v) for v in values]
                stacked = np.stack(arrs, axis=1)  # (n, C, F)
                if class_idx < 0 or class_idx >= stacked.shape[1]:
                    raise ValueError(f"class_idx={class_idx} out of bounds for {stacked.shape[1]} classes")
                return stacked[:, class_idx, :], stacked.shape[1]
            vals = np.asarray(values)
            if vals.ndim == 3:
                n, a1, a2 = vals.shape
                if a2 == n_features:          # (n, C, F)
                    C = a1
                    if class_idx < 0 or class_idx >= C:
                        raise ValueError(f"class_idx={class_idx} out of bounds for {C} classes")
                    return vals[:, class_idx, :], C
                if a1 == n_features:          # (n, F, C)  <-- your case
                    C = a2
                    if class_idx < 0 or class_idx >= C:
                        raise ValueError(f"class_idx={class_idx} out of bounds for {C} classes")
                    return vals[:, :, class_idx], C
                raise RuntimeError(f"Unexpected SHAP shape {vals.shape}: neither axis equals n_features={n_features}.")
            if vals.ndim == 2:
                if vals.shape[1] == n_features:
                    return vals, None
                if vals.shape[1] % n_features == 0:
                    C = vals.shape[1] // n_features
                    reshaped = vals.reshape(vals.shape[0], C, n_features)
                    if class_idx < 0 or class_idx >= C:
                        raise ValueError(f"class_idx={class_idx} out of bounds for {C} classes")
                    return reshaped[:, class_idx, :], C
                raise RuntimeError(
                    f"SHAP values have {vals.shape[1]} columns, not equal to n_features={n_features} "
                    f"and not a multiple thereof."
                )
            raise RuntimeError(f"Unsupported SHAP values ndim={vals.ndim}")

        def pick_base_values(base_vals, n_classes_detected, class_idx):
            import numpy as np
            if n_classes_detected in (None, 1):
                return base_vals
            bv = np.asarray(base_vals)
            if bv.ndim == 0:
                return float(bv)
            if bv.ndim == 1 and bv.size == n_classes_detected:
                return bv[class_idx]
            if bv.ndim == 2 and bv.shape[1] == n_classes_detected:
                return bv[:, class_idx]
            return base_vals

        # --- dirs ---
        out_dir = os.path.join(experiment_dir, "figure", "shap")
        os.makedirs(out_dir, exist_ok=True)

        # --- unpack pipeline / estimator ---
        if isinstance(model_pipeline, Pipeline):
            if not model_pipeline.steps:
                raise ValueError("Pipeline has no steps.")
            estimator = model_pipeline.steps[-1][1]
            preprocess = None
            for name, step in model_pipeline.steps[:-1]:
                if name.lower() in ("preprocess", "preprocessing", "preproc"):
                    preprocess = step
                    break
            if preprocess is None:
                for name, step in model_pipeline.steps[:-1]:
                    if hasattr(step, "transform"):
                        preprocess = step
                        break
        else:
            estimator = model_pipeline
            preprocess = None

        # --- transform X (skip SMOTE) ---
        if preprocess is not None and hasattr(preprocess, "transform"):
            X_trans = preprocess.transform(X)
            feature_names = None
            if hasattr(preprocess, "get_feature_names_out"):
                try:
                    feature_names = list(preprocess.get_feature_names_out())
                except Exception:
                    feature_names = None
        else:
            X_trans = X
            feature_names = None

        # --- DataFrame ---
        if isinstance(X_trans, pd.DataFrame):
            X_df = X_trans.copy()
        else:
            n_cols = getattr(X_trans, "shape", (0, 0))[1]
            if feature_names is None and isinstance(X, pd.DataFrame):
                feature_names = [str(c) for c in X.columns]
            if feature_names is not None and len(feature_names) == n_cols:
                X_df = pd.DataFrame(X_trans, columns=feature_names, index=getattr(X, "index", None))
            else:
                X_df = pd.DataFrame(X_trans, columns=[f"f{i}" for i in range(n_cols)], index=getattr(X, "index", None))

        if sparse.issparse(X_df.values):
            X_df = pd.DataFrame(X_df.values.todense(), columns=X_df.columns, index=X_df.index)

        # --- estimator input sanity ---
        n_in = getattr(estimator, "n_features_in_", None)
        if n_in is not None and n_in != X_df.shape[1]:
            raise RuntimeError(
                f"Estimator was trained on {n_in} features but preprocessed X has {X_df.shape[1]}."
            )

        # --- subsample ---
        X_sample = X_df.sample(sample_size, random_state=42) if len(X_df) > sample_size else X_df
        n_feat = X_sample.shape[1]

        # --- compute SHAP ---
        try:
            if is_tree_model(estimator):
                explainer = shap.TreeExplainer(estimator, model_output="raw")
                shap_vals = explainer.shap_values(X_sample.values)
                base_vals = explainer.expected_value
            else:
                if hasattr(estimator, "predict_proba"):
                    f = lambda data: estimator.predict_proba(data)
                elif hasattr(estimator, "decision_function"):
                    f = lambda data: estimator.decision_function(data)
                else:
                    f = lambda data: estimator.predict(data)
                explainer = shap.Explainer(f, X_sample.values, feature_names=X_sample.columns.tolist())
                expl = explainer(X_sample.values)
                shap_vals = expl.values
                base_vals = expl.base_values
        except Exception as e:
            raise RuntimeError(f"Failed to compute SHAP values: {e}")

        # --- normalize to (n, n_features) ---
        values_2d, n_classes_detected = normalize_shap_values(shap_vals, n_feat, class_idx)
        base_vals = pick_base_values(base_vals, n_classes_detected, class_idx)

        if values_2d.shape[1] != n_feat:
            raise RuntimeError(
                f"After normalization, SHAP has {values_2d.shape[1]} features but X has {n_feat}."
            )

        # --- PRETTIFY feature names (NEW) ---
        if feature_namer is not None:
            pretty_cols = [feature_namer(c) for c in X_sample.columns]
        else:
            pretty_cols = list(X_sample.columns)

        # --- Explanation & plots ---
        expl_for_plot = shap.Explanation(
            values=values_2d,
            base_values=base_vals,
            data=X_sample.values,
            feature_names=pretty_cols,   # <-- use pretty names in plots
        )

        # Importance table (pretty names)
        mean_abs = np.mean(np.abs(values_2d), axis=0)
        mean_abs_1d = np.asarray(mean_abs).reshape(-1)
        features_1d = pretty_cols

        importance = (
            pd.DataFrame({"feature": features_1d, "mean_abs_shap": mean_abs_1d}, copy=False)
            .sort_values("mean_abs_shap", ascending=False)
            .reset_index(drop=True)
        )

        if getattr(self, "verbose", False):
            self.printer(f"[Evaluator] X_sample shape: {X_sample.shape}\n")
            self.printer(f"[Evaluator] SHAP values (normalized) shape: {values_2d.shape}\n")
            self.printer("[Evaluator] Top features by |SHAP|:\n")
            self.printer(importance.head(min(10, len(importance))).to_string(index=False) + "\n")

        return importance


    def make_stai_feature_namer(self, metadata_df):
        """
        Build a callable that converts engineered column names into readable labels.

        Parameters
        ----------
        metadata_df : pd.DataFrame
            Must contain columns:
            - 'variable' (e.g., 'stai_40_tp1', 'stai_state_score_tp1')
            - 'label'    (human-readable question text or variable label)
            - 'choices'  (optional; e.g., '1 = Not at all; 2 = Somewhat; 3 = Moderately; 4 = Very much')

        Returns
        -------
        namer : callable
            namer(raw_col_name: str) -> pretty_label: str
        """
        import re

        # Normalize column names in metadata
        md = metadata_df.copy()
        md['variable'] = md['variable'].astype(str)

        # Parse the choices column (to map e.g. 2 -> 'Somewhat')
        def _parse_choices(s):
            if not isinstance(s, str) or s.strip() == "":
                return {}
            # split on ; and map "N = text"
            out = {}
            for part in s.split(';'):
                part = part.strip()
                if not part:
                    continue
                # match "number = text"
                m = re.match(r'^(-?\d+(?:\.\d+)?)\s*=\s*(.+)$', part)
                if m:
                    k = m.group(1)
                    # keep both '2' and '2.0' keys (OneHotEncoder often appends '.0')
                    out[k] = m.group(2).strip()
                    if k.endswith('.0'):
                        out[k[:-2]] = out[k]
                    else:
                        out[k + '.0'] = out[k]
            return out

        md['choice_map'] = md.get('choices', "").apply(_parse_choices)

        # Build lookup dicts
        var_to_label     = dict(zip(md['variable'], md['label']))
        var_to_choiceMap = dict(zip(md['variable'], md['choice_map']))

        # Regex to detect one-hot columns like "<base>_<level>" where level is a number (possibly float)
        onehot_pat = re.compile(r'^(?P<base>.+)_(?P<lvl>-?\d+(?:\.\d+)?)$')

        def namer(col: str) -> str:
            """
            Convert engineered col name to a readable label.
            - stai_40_tp1_2.0 -> "Q40: <label> = Somewhat"
            - stai_06_tp1_1.0 -> "Q06: <label> = Not at all"
            - stai_state_score_tp1 -> "<label>" (no level)
            If metadata is missing, fall back gracefully to the raw name.
            """
            m = onehot_pat.match(col)
            if m:
                base = m.group('base')
                lvl  = m.group('lvl')
                base_label = var_to_label.get(base)
                choice_map = var_to_choiceMap.get(base, {})
                # Try to infer question number like stai_40_tp1 -> Q40
                qnum = None
                qmatch = re.search(r'stai_(\d+)_tp\d+', base)
                if qmatch:
                    qnum = qmatch.group(1).zfill(2)  # "40" -> "40", "3" -> "03"
                lvl_text = choice_map.get(lvl) or choice_map.get(lvl.rstrip('0').rstrip('.'))  # extra safety

                if base_label and lvl_text:
                    if qnum:
                        return f"Q{qnum}: {base_label}; Ans: {lvl_text}"
                    return f"{base_label} = {lvl_text}"
                # Partial metadata -> degrade gracefully
                if base_label:
                    if qnum:
                        return f"Q{qnum}: {base_label} = {lvl}"
                    return f"{base_label} = {lvl}"
                # No metadata
                return col
            else:
                # Not a one-hot pattern; map base variable label if we have it
                base_label = var_to_label.get(col)
                return base_label if base_label else col

        return namer

    def save_shap_importance_summary(
        self,
        all_imps_df,
        experiment_dir: str = "results_unknown_date",
        top_k: int = 30,
        ci: float = 0.95,
        figsize=(12, 6),
        filename_stub: str = "shap_importance_mean_std",
    ):
        """
        Aggregate SHAP importances across iterations, save CSVs, and plot.

        Parameters
        ----------
        all_imps_df : pd.DataFrame
            Long-form DF with columns at least: ['feature', 'mean_abs_shap', 'iter'].
        experiment_dir : str
            Root directory for outputs. CSVs -> <experiment_dir>/table/shap/,
            figures -> <experiment_dir>/figure/shap/.
        top_k : int
            Number of top features (by mean importance) to show in the plot.
        ci : float
            Confidence level for mean CI bars (e.g., 0.95 -> 95% CI).
        figsize : tuple
            Matplotlib figure size for the bar plot.
        filename_stub : str
            Base filename (without extension) for outputs.

        Returns
        -------
        dict
            {
            "summary_df": <pd.DataFrame>,
            "csv_path": <str>,
            "csv_long_path": <str>,
            "png_path": <str>,
            "pdf_path": <str>
            }
        """
        import os
        import math
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        from scipy.stats import norm

        # --- validate input ---
        required_cols = {"feature", "mean_abs_shap", "iter"}
        missing = required_cols.difference(set(all_imps_df.columns))
        if missing:
            raise ValueError(f"all_imps_df is missing required columns: {sorted(missing)}")

        # --- dirs ---
        fig_dir = os.path.join(experiment_dir, "figure", "shap")
        tab_dir = os.path.join(experiment_dir, "table", "shap")
        os.makedirs(fig_dir, exist_ok=True)
        os.makedirs(tab_dir, exist_ok=True)

        # --- aggregate ---
        grouped = (
            all_imps_df
            .groupby("feature", as_index=False)["mean_abs_shap"]
            .agg(mean="mean", std="std", n="count")
        )

        # handle potential 0 or NaN std (e.g., single iteration)
        grouped["std"] = grouped["std"].fillna(0.0)

        # coefficient of variation
        grouped["cv"] = grouped["std"] / grouped["mean"].replace(0, np.nan)

        # confidence interval for the mean (normal approx)
        alpha = 1.0 - ci
        z = norm.ppf(1 - alpha / 2.0)
        grouped["se"] = grouped["std"] / grouped["n"].clip(lower=1).pow(0.5)
        grouped["mean_ci_lo"] = grouped["mean"] - z * grouped["se"]
        grouped["mean_ci_hi"] = grouped["mean"] + z * grouped["se"]

        # sort by mean descending
        summary_df = grouped.sort_values("mean", ascending=False).reset_index(drop=True)

        # --- save CSVs ---
        csv_path = os.path.join(tab_dir, f"{filename_stub}.csv")
        csv_long_path = os.path.join(tab_dir, f"{filename_stub}_all_iters_long.csv")

        # tidy summary
        summary_df.to_csv(csv_path, index=False)

        # also save the long data (as provided), sorted for convenience
        all_imps_df.sort_values(["feature", "iter"]).to_csv(csv_long_path, index=False)

        # --- plot top_k with error bars (mean ± 95% CI by default) ---
        top = summary_df.head(top_k).copy()
        # plot in descending order left→right
        top = top.iloc[::-1]  # reverse so the largest appears at the top in horizontal bar

        plt.figure(figsize=figsize)
        y_positions = np.arange(len(top))

        # horizontal bars: mean
        plt.barh(y_positions, top["mean"].values)

        # error bars: CI
        xerr = np.vstack([top["mean"].values - top["mean_ci_lo"].values,
                        top["mean_ci_hi"].values - top["mean"].values])
        plt.errorbar(
            top["mean"].values,
            y_positions,
            xerr=xerr,
            fmt="none",
            capsize=3,
            linewidth=1,
        )

        plt.yticks(y_positions, top["feature"].values)
        plt.xlabel("Mean |SHAP value| across iterations")
        plt.title(f"Top {min(top_k, len(summary_df))} features by mean |SHAP| (with {int(ci*100)}% CI)")
        plt.tight_layout()

        png_path = os.path.join(fig_dir, f"{filename_stub}.png")
        pdf_path = os.path.join(fig_dir, f"{filename_stub}.pdf")
        plt.savefig(png_path, dpi=200)
        plt.savefig(pdf_path)
        plt.close()

        if getattr(self, "verbose", False):
            self.printer(f"[Evaluator] Saved SHAP summary table to: {csv_path}\n")
            self.printer(f"[Evaluator] Saved long-form iterations to: {csv_long_path}\n")
            self.printer(f"[Evaluator] Saved SHAP summary plot to: {png_path}\n")

        return {
            "summary_df": summary_df,
            "csv_path": csv_path,
            "csv_long_path": csv_long_path,
            "png_path": png_path,
            "pdf_path": pdf_path,
        }

    def collect_roc_pr_curves(
        self,
        model_pipeline,
        X,
        y,
        class_idx: int = 1,
        experiment_dir: str = "results_unknown_date",
        n_points: int = 1001,
        run_name: str = None,
    ):
        """
        Compute ROC and PR curves for a binary target using the given model/pipeline.
        Interpolates curves onto fixed grids so multiple runs can be aggregated later.
        Appends results to an on-disk accumulator (.npz) and returns the run metrics.

        Notes
        -----
        - Positive class is assumed to be label 1 in y. If your labels are different,
        remap before calling.
        - For binary models, the function auto-selects column 1 of predict_proba (or
        uses decision_function directly), regardless of class_idx.
        """
        import os
        import numpy as np
        import pandas as pd
        from sklearn.pipeline import Pipeline
        from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

        # --- dirs / files ---
        curve_dir = os.path.join(experiment_dir, "figure", "curves")
        os.makedirs(curve_dir, exist_ok=True)
        agg_path = os.path.join(curve_dir, "curves_accumulator.npz")
        runs_csv = os.path.join(curve_dir, "per_run_metrics.csv")

        # --- unpack estimator (not strictly needed, but kept for clarity) ---
        estimator = model_pipeline
        if isinstance(model_pipeline, Pipeline) and model_pipeline.steps:
            estimator = model_pipeline.steps[-1][1]

        # --- decide which score vector to use ---
        # Binary safeguard: if predict_proba has 2 columns, always use column 1.
        scores = None
        if hasattr(model_pipeline, "predict_proba"):
            proba = model_pipeline.predict_proba(X)
            if proba.ndim == 2 and proba.shape[1] == 2:
                # binary → use positive class probability
                scores = proba[:, 1]
            elif proba.ndim == 2 and proba.shape[1] > 2:
                # multiclass OvR-style → honor class_idx
                scores = proba[:, class_idx]
            else:
                scores = np.ravel(proba)
        if scores is None and hasattr(model_pipeline, "decision_function"):
            df = model_pipeline.decision_function(X)
            if df.ndim == 2:  # OvR/OVO scores
                if df.shape[1] == 1:
                    scores = np.ravel(df)
                else:
                    # if truly multiclass, pick class_idx; if binary but 2D, pick column 1
                    use_idx = 1 if df.shape[1] == 2 else class_idx
                    scores = df[:, use_idx]
            else:
                scores = np.ravel(df)
        if scores is None:
            # Fallback: labels (not ideal, but defined)
            scores = np.ravel(model_pipeline.predict(X))

        # --- raw curves / areas (assumes positive label = 1) ---
        fpr, tpr, _ = roc_curve(y_true=y, y_score=scores, pos_label=1)
        roc_auc = auc(fpr, tpr)

        precision, recall, _ = precision_recall_curve(y_true=y, probas_pred=scores, pos_label=1)
        ap = average_precision_score(y_true=y, y_score=scores, pos_label=1)

        # --- interpolation grids (fixed for aggregation) ---
        fpr_grid = np.linspace(0.0, 1.0, n_points)
        # ROC: interpolate TPR vs FPR
        tpr_interp = np.interp(fpr_grid, fpr, tpr)
        tpr_interp[0] = 0.0
        tpr_interp[-1] = 1.0

        # PR: sklearn returns recall increasing from 0→1; ensure sorted for interp
        order = np.argsort(recall)
        recall_sorted = recall[order]
        precision_sorted = precision[order]
        # Make precision a non-increasing function of recall to reduce wiggles
        precision_envelope = np.maximum.accumulate(precision_sorted[::-1])[::-1]

        recall_grid = np.linspace(0.0, 1.0, n_points)
        prec_interp = np.interp(recall_grid, recall_sorted, precision_envelope)
        prec_interp = np.clip(prec_interp, 0.0, 1.0)

        # --- persist to accumulator (no object dtype; keep allow_pickle=False) ---
        # run_names stored as fixed-width Unicode to avoid object arrays
        this_name = run_name if run_name is not None else "run_0"

        if os.path.exists(agg_path):
            data = np.load(agg_path, allow_pickle=False)
            same_grid = (
                np.allclose(data["fpr_grid"], fpr_grid)
                and np.allclose(data["recall_grid"], recall_grid)
                and int(data["n_points"]) == int(n_points)
            )
            if not same_grid:
                # Start fresh if grids differ
                tpr_stack = tpr_interp[None, :]
                prec_stack = prec_interp[None, :]
                roc_aucs = np.array([roc_auc], dtype=float)
                aps = np.array([ap], dtype=float)
                run_names = np.array([this_name], dtype="U256")
            else:
                tpr_stack = np.vstack([data["tpr_stack"], tpr_interp[None, :]])
                prec_stack = np.vstack([data["prec_stack"], prec_interp[None, :]])
                roc_aucs = np.append(data["roc_aucs"], roc_auc)
                aps = np.append(data["aps"], ap)
                prev_names = data["run_names"].astype("U256")
                if run_name is None:
                    this_name = f"run_{prev_names.size}"
                run_names = np.append(prev_names, np.array([this_name], dtype="U256"))
        else:
            tpr_stack = tpr_interp[None, :]
            prec_stack = prec_interp[None, :]
            roc_aucs = np.array([roc_auc], dtype=float)
            aps = np.array([ap], dtype=float)
            run_names = np.array([this_name], dtype="U256")

        np.savez_compressed(
            agg_path,
            fpr_grid=fpr_grid,
            recall_grid=recall_grid,
            tpr_stack=tpr_stack,
            prec_stack=prec_stack,
            roc_aucs=roc_aucs,
            aps=aps,
            run_names=run_names,   # dtype '<U...' (NOT object)
            n_points=int(n_points),
        )

        # Append per-run summary CSV (easy to inspect later)
        row = pd.DataFrame(
            [{
                "run_name": this_name,
                "roc_auc": roc_aucs[-1],
                "average_precision": aps[-1],
                "n_points": n_points,
                "n_obs": len(y),
            }]
        )
        if os.path.exists(runs_csv):
            row.to_csv(runs_csv, mode="a", header=False, index=False)
        else:
            row.to_csv(runs_csv, index=False)

        if getattr(self, "verbose", False):
            self.printer(
                f"[Curves] ROC AUC={roc_auc:.5f} | AP={ap:.5f} | "
                f"accumulator: {agg_path} | per-run metrics: {runs_csv}\n"
            )

        return {
            "fpr": fpr, "tpr": tpr, "roc_auc": roc_auc,
            "recall": recall, "precision": precision, "average_precision": ap,
            "fpr_grid": fpr_grid, "tpr_interp": tpr_interp,
            "recall_grid": recall_grid, "prec_interp": prec_interp,
            "run_name": this_name,
            "n_points": n_points,
            "accumulator": agg_path,
            "runs_csv": runs_csv,
            "out_dir": curve_dir,
        }



    # def plot_aggregated_curves(
    #     self,
    #     experiment_dir: str = "results_unknown_date",
    #     ci: float = 0.95,
    #     save_csv: bool = True,
    #     dpi: int = 180,
    # ):
    #     """
    #     Load the accumulator and plot mean ± std and 95% CI bands for ROC and PR.
    #     Saves PNG+SVG and (optionally) CSVs into <experiment_dir>/figure/curves/.
    #     """
    #     import os
    #     import numpy as np
    #     import pandas as pd
    #     import matplotlib.pyplot as plt
    #     from math import sqrt
    #     from scipy.stats import norm

    #     curve_dir = os.path.join(experiment_dir, "figure", "curves")
    #     agg_path = os.path.join(curve_dir, "curves_accumulator.npz")
    #     os.makedirs(curve_dir, exist_ok=True)

    #     if not os.path.exists(agg_path):
    #         raise FileNotFoundError(
    #             f"No accumulator found at {agg_path}. "
    #             "Run `collect_roc_pr_curves` in your iterations first."
    #         )

    #     data = np.load(agg_path, allow_pickle=False)
    #     fpr_grid = data["fpr_grid"]
    #     recall_grid = data["recall_grid"]
    #     tpr_stack = data["tpr_stack"]  # (R, P)
    #     prec_stack = data["prec_stack"]  # (R, P)
    #     roc_aucs = data["roc_aucs"]
    #     aps = data["aps"]
    #     run_names = data["run_names"]
    #     n_runs = tpr_stack.shape[0]
    #     z = norm.ppf(0.5 + ci / 2.0)

    #     # --- stats on grids ---
    #     def mean_std_ci(arr_stack):
    #         mu = np.nanmean(arr_stack, axis=0)
    #         sd = np.nanstd(arr_stack, axis=0, ddof=1) if n_runs > 1 else np.zeros_like(mu)
    #         se = sd / sqrt(n_runs) if n_runs > 1 else np.zeros_like(mu)
    #         lo = np.clip(mu - z * se, 0.0, 1.0)
    #         hi = np.clip(mu + z * se, 0.0, 1.0)
    #         return mu, sd, lo, hi

    #     tpr_mean, tpr_sd, tpr_lo, tpr_hi = mean_std_ci(tpr_stack)
    #     prec_mean, prec_sd, prec_lo, prec_hi = mean_std_ci(prec_stack)

    #     # --- summary tables ---
    #     summary = {
    #         "n_runs": int(n_runs),
    #         "roc_auc_mean": float(np.mean(roc_aucs)),
    #         "roc_auc_std": float(np.std(roc_aucs, ddof=1) if n_runs > 1 else 0.0),
    #         "ap_mean": float(np.mean(aps)),
    #         "ap_std": float(np.std(aps, ddof=1) if n_runs > 1 else 0.0),
    #     }

    #     if save_csv:
    #         pd.DataFrame({
    #             "fpr": fpr_grid,
    #             "tpr_mean": tpr_mean,
    #             "tpr_std": tpr_sd,
    #             "tpr_lo": tpr_lo,
    #             "tpr_hi": tpr_hi,
    #         }).to_csv(os.path.join(curve_dir, "roc_mean_band.csv"), index=False)

    #         pd.DataFrame({
    #             "recall": recall_grid,
    #             "precision_mean": prec_mean,
    #             "precision_std": prec_sd,
    #             "precision_lo": prec_lo,
    #             "precision_hi": prec_hi,
    #         }).to_csv(os.path.join(curve_dir, "pr_mean_band.csv"), index=False)

    #         pd.DataFrame({
    #             "run_name": run_names,
    #             "roc_auc": roc_aucs,
    #             "average_precision": aps,
    #         }).to_csv(os.path.join(curve_dir, "per_run_auc_ap.csv"), index=False)

    #     # --- plotting style (vivid & readable) ---
    #     plt.rcParams.update({
    #         "figure.figsize": (7.5, 6.0),
    #         "axes.spines.top": False,
    #         "axes.spines.right": False,
    #         "axes.labelsize": 12,
    #         "axes.titlesize": 14,
    #         "legend.frameon": False,
    #         "grid.alpha": 0.25,
    #         "lines.linewidth": 2.5,
    #     })

    #     # --- ROC plot ---
    #     fig1, ax1 = plt.subplots()
    #     ax1.plot([0, 1], [0, 1], linestyle="--", linewidth=1.5, alpha=0.8, label="Chance")
    #     ax1.plot(fpr_grid, tpr_mean, label=f"Mean ROC (AUC={summary['roc_auc_mean']:.3f})")
    #     ax1.fill_between(fpr_grid, tpr_lo, tpr_hi, alpha=0.2, label=f"{int(ci*100)}% CI")
    #     # add ±1 SD band (lighter)
    #     ax1.fill_between(fpr_grid, np.clip(tpr_mean - tpr_sd, 0, 1), np.clip(tpr_mean + tpr_sd, 0, 1),
    #                     alpha=0.12, label="±1 SD")
    #     ax1.set_title(f"ROC Curve — runs={n_runs}")
    #     ax1.set_xlabel("False Positive Rate")
    #     ax1.set_ylabel("True Positive Rate")
    #     ax1.grid(True, linestyle=":")
    #     ax1.legend()
    #     roc_png = os.path.join(curve_dir, "roc_mean_ci.png")
    #     roc_svg = os.path.join(curve_dir, "roc_mean_ci.svg")
    #     fig1.tight_layout()
    #     fig1.savefig(roc_png, dpi=dpi)
    #     fig1.savefig(roc_svg)
    #     plt.close(fig1)

    #     # --- PR plot ---
    #     fig2, ax2 = plt.subplots()
    #     ax2.plot(recall_grid, prec_mean, label=f"Mean PR (AP={summary['ap_mean']:.3f})")
    #     ax2.fill_between(recall_grid, prec_lo, prec_hi, alpha=0.2, label=f"{int(ci*100)}% CI")
    #     ax2.fill_between(recall_grid,
    #                     np.clip(prec_mean - prec_sd, 0, 1),
    #                     np.clip(prec_mean + prec_sd, 0, 1),
    #                     alpha=0.12, label="±1 SD")
    #     ax2.set_title(f"Precision–Recall — runs={n_runs}")
    #     ax2.set_xlabel("Recall")
    #     ax2.set_ylabel("Precision")
    #     ax2.set_xlim(0, 1)
    #     ax2.set_ylim(0, 1)
    #     ax2.grid(True, linestyle=":")
    #     ax2.legend()
    #     pr_png = os.path.join(curve_dir, "pr_mean_ci.png")
    #     pr_svg = os.path.join(curve_dir, "pr_mean_ci.svg")
    #     fig2.tight_layout()
    #     fig2.savefig(pr_png, dpi=dpi)
    #     fig2.savefig(pr_svg)
    #     plt.close(fig2)

    #     if getattr(self, "verbose", False):
    #         self.printer(
    #             "[Curves] Aggregated plots saved:\n"
    #             f"  ROC: {roc_png}\n"
    #             f"  PR : {pr_png}\n"
    #             f"  (SVGs also saved)\n"
    #             f"  Summary: runs={n_runs}, AUC mean±std={summary['roc_auc_mean']:.4f}±{summary['roc_auc_std']:.4f}, "
    #             f"AP mean±std={summary['ap_mean']:.4f}±{summary['ap_std']:.4f}\n"
    #         )

    #     return {
    #         "summary": summary,
    #         "roc_png": roc_png,
    #         "pr_png": pr_png,
    #         "accumulator": agg_path,
    #         "out_dir": curve_dir,
    #     }

    def plot_aggregated_curves(
        self,
        experiment_dir: str = "results_unknown_date",
        ci: float = 0.95,
        save_csv: bool = True,
        dpi: int = 180,
        smooth_lpf: bool = False,
        lpf_cutoff: float = 0.009,
        lpf_order: int = 2,
        font_size: float = 12.0,
        filter_type: str = "box",   # "butter" or "box"
        box_window: int =  500,  # moving-average window when filter_type="box"
        show_ci: bool = True,
        show_std: bool = False,
        pr_chance: bool = True,
    ):
        """
        Load accumulator and plot mean ± std and 95% CI bands for ROC and PR.
        Saves PNG+SVG and (optionally) CSVs into <experiment_dir>/figure/curves/.

        Guarantees (by construction):
        • ROC curves pass through (0,0) and (1,1).
        • PR curves pass through (0,1) and (1,0).

        Smoothing:
        - If `smooth_lpf=True` and filter_type == "butter": zero-phase Butterworth LPF.
        - If `smooth_lpf=True` and filter_type == "box": zero-phase moving-average (boxcar) via filtfilt.
        Anchors are ADDED TO THE SIGNAL BEFORE FILTERING, then endpoints are re-pinned after filtering.
        """
        import os
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        from math import sqrt
        from scipy.stats import norm
        from scipy.signal import butter, filtfilt

        # ---------------- I/O ----------------
        # Path to per-run performance CSV
        perf_csv_path = os.path.join(experiment_dir, "table", "performance", "per_run_performance.csv")
        if os.path.exists(perf_csv_path):
            perf_df = pd.read_csv(perf_csv_path)
            positive_class_ratio = perf_df["Prevalence"].mean()
            positive_rate = perf_df["Prevalence"].mean()  # mean prevalence across runs
        else:
            positive_class_ratio = None
            print(f"Warning: Performance CSV not found at {perf_csv_path}. PR chance line may be incorrect.")


        curve_dir = os.path.join(experiment_dir, "figure", "curves")
        agg_path = os.path.join(curve_dir, "curves_accumulator.npz")
        os.makedirs(curve_dir, exist_ok=True)
        if not os.path.exists(agg_path):
            raise FileNotFoundError(
                f"No accumulator found at {agg_path}. "
                "Run `collect_roc_pr_curves` in your iterations first."
            )

        data = np.load(agg_path, allow_pickle=False)
        fpr_grid = np.asarray(data["fpr_grid"], dtype=float)          # (P_roc,)
        recall_grid = np.asarray(data["recall_grid"], dtype=float)    # (P_pr,)
        tpr_stack = np.asarray(data["tpr_stack"], dtype=float)        # (R, P_roc)
        prec_stack = np.asarray(data["prec_stack"], dtype=float)      # (R, P_pr)
        roc_aucs = np.asarray(data["roc_aucs"], dtype=float)
        aps = np.asarray(data["aps"], dtype=float)
        run_names = np.asarray(data["run_names"])
        n_runs = tpr_stack.shape[0]
        z = norm.ppf(0.5 + ci / 2.0)

        # ---------------- helpers ----------------
        def _naninterp1d(y: np.ndarray) -> np.ndarray:
            """Linear interpolate NaNs; all-NaN -> zeros; single valid -> flat."""
            y = np.asarray(y, dtype=float)
            if np.all(np.isnan(y)):
                return np.zeros_like(y)
            x = np.arange(y.size, dtype=float)
            m = ~np.isnan(y)
            if m.sum() == 1:
                return np.full_like(y, y[m][0])
            return np.interp(x, x[m], y[m])

        def _butter_lpf(y: np.ndarray, cutoff: float, order: int) -> np.ndarray:
            """Zero-phase Butterworth LPF with safe padding; clip to [0,1]."""
            y = _naninterp1d(y)
            Wn = float(np.clip(cutoff, 1e-4, 0.99))  # normalized (0,1)
            b, a = butter(order, Wn, btype="low", analog=False)
            default_pad = 3 * max(len(a), len(b))
            padlen = min(default_pad, max(0, y.size - 2))
            try:
                yf = filtfilt(b, a, y, padlen=padlen)
            except ValueError:
                yf = filtfilt(b, a, y, padlen=0)
            return np.clip(yf, 0.0, 1.0)

        def _boxcar_lpf(y: np.ndarray, window: int) -> np.ndarray:
            """Zero-phase moving-average smoothing via filtfilt; clip to [0,1]."""
            y = _naninterp1d(y)
            n = y.size
            if n < 3:
                return np.clip(y, 0.0, 1.0)
            w = int(max(3, window))
            # keep window feasible relative to signal and filtfilt padding
            w = min(w, max(3, n - 1))
            # make odd for symmetry (optional but nice)
            if w % 2 == 0:
                w = max(3, w - 1)
            b = np.ones(w, dtype=float) / w
            a = np.array([1.0], dtype=float)
            default_pad = 3 * max(len(a), len(b))
            padlen = min(default_pad, max(0, n - 2))
            try:
                yf = filtfilt(b, a, y, padlen=padlen)
            except ValueError:
                yf = filtfilt(b, a, y, padlen=0)
            return np.clip(yf, 0.0, 1.0)

        def _inject_anchors_and_interp(x_old, Y_old, anchors):
            """
            Ensure x contains 0 and 1 (via union) and Y is interpolated to x_new.
            Then pin the anchor columns to the desired values.
            anchors: list of (x_anchor, y_value) pairs at endpoints and/or midpoints.
            Returns x_new, Y_new.
            """
            x_new = np.union1d(x_old, np.array([0.0, 1.0], dtype=float))  # sorted unique
            Y_new = np.empty((Y_old.shape[0], x_new.size), dtype=float)
            for i in range(Y_old.shape[0]):
                Y_new[i] = np.interp(x_new, x_old, _naninterp1d(Y_old[i]))
            # Pin anchors
            for x_anchor, y_val in anchors:
                if np.isclose(x_anchor, 0.0):
                    Y_new[:, 0] = y_val
                elif np.isclose(x_anchor, 1.0):
                    Y_new[:, -1] = y_val
                else:
                    j = int(np.argmin(np.abs(x_new - x_anchor)))
                    if np.isclose(x_new[j], x_anchor):
                        Y_new[:, j] = y_val
            return x_new, Y_new

        # ---------------- include anchors BEFORE filtering ----------------
        # ROC wants (0,0) and (1,1)
        fpr_grid, tpr_stack = _inject_anchors_and_interp(
            fpr_grid, tpr_stack, anchors=[(0.0, 0.0), (1.0, 1.0)]
        )
        # PR wants (0,1) and (1,positive_rate)
        recall_grid, prec_stack = _inject_anchors_and_interp(
            recall_grid, prec_stack, anchors=[(0.0, 1), (1.0, positive_rate)]
        )

        # ---------------- optional smoothing ----------------
        if smooth_lpf:
            if filter_type.lower() == "box":
                for i in range(n_runs):
                    tpr_stack[i] = _boxcar_lpf(tpr_stack[i], window=box_window)
                    prec_stack[i] = _boxcar_lpf(prec_stack[i], window=box_window)
                smoothing_note = f" (Box, w={box_window})"
            else:  # default to butter
                for i in range(n_runs):
                    tpr_stack[i] = _butter_lpf(tpr_stack[i], cutoff=lpf_cutoff, order=lpf_order)
                    prec_stack[i] = _butter_lpf(prec_stack[i], cutoff=lpf_cutoff, order=lpf_order)
                smoothing_note = f" (Butter, fc={lpf_cutoff}, ord={lpf_order})"
            # Re-pin endpoints AFTER filtering to ensure exact anchors
            tpr_stack[:, 0] = 0.0
            tpr_stack[:, -1] = 1.0
            prec_stack[:, 0] = 1.0
            prec_stack[:, -1] = positive_rate
        else:
            smoothing_note = ""

        # ---------------- stats ----------------
        def mean_std_ci(arr_stack):
            mu = np.nanmean(arr_stack, axis=0)
            sd = np.nanstd(arr_stack, axis=0, ddof=1) if n_runs > 1 else np.zeros_like(mu)
            se = sd / sqrt(n_runs) if n_runs > 1 else np.zeros_like(mu)
            lo = np.clip(mu - z * se, 0.0, 1.0)
            hi = np.clip(mu + z * se, 0.0, 1.0)
            return mu, sd, lo, hi

        tpr_mean, tpr_sd, tpr_lo, tpr_hi = mean_std_ci(tpr_stack)
        prec_mean, prec_sd, prec_lo, prec_hi = mean_std_ci(prec_stack)

        summary = {
            "n_runs": int(n_runs),
            "roc_auc_mean": float(np.mean(roc_aucs)),
            "roc_auc_std": float(np.std(roc_aucs, ddof=1) if n_runs > 1 else 0.0),
            "ap_mean": float(np.mean(aps)),
            "ap_std": float(np.std(aps, ddof=1) if n_runs > 1 else 0.0),
            "smoothed": bool(smooth_lpf),
            "filter_type": filter_type.lower() if smooth_lpf else None,
            "lpf_cutoff": float(lpf_cutoff) if (smooth_lpf and filter_type.lower() == "butter") else None,
            "lpf_order": int(lpf_order) if (smooth_lpf and filter_type.lower() == "butter") else None,
            "box_window": int(box_window) if (smooth_lpf and filter_type.lower() == "box") else None,
        }

        # ---------------- CSVs ----------------
        suffix = "_smooth" if smooth_lpf else ""
        if save_csv:
            pd.DataFrame({
                "fpr": fpr_grid,
                "tpr_mean": tpr_mean,
                "tpr_std": tpr_sd,
                "tpr_lo": tpr_lo,
                "tpr_hi": tpr_hi,
            }).to_csv(os.path.join(curve_dir, f"roc_mean_band{suffix}.csv"), index=False)

            pd.DataFrame({
                "recall": recall_grid,
                "precision_mean": prec_mean,
                "precision_std": prec_sd,
                "precision_lo": prec_lo,
                "precision_hi": prec_hi,
            }).to_csv(os.path.join(curve_dir, f"pr_mean_band{suffix}.csv"), index=False)

            pd.DataFrame({
                "run_name": run_names,
                "roc_auc": roc_aucs,
                "average_precision": aps,
            }).to_csv(os.path.join(curve_dir, f"per_run_auc_ap{suffix}.csv"), index=False)

        # ---------------- styling (uniform font size) ----------------
        plt.rcParams.update({
            "figure.figsize": (7.5, 6.0),
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": font_size,
            "axes.labelsize": font_size,
            "axes.titlesize": font_size,
            "legend.fontsize": font_size,
            "xtick.labelsize": font_size,
            "ytick.labelsize": font_size,
            "grid.alpha": 0.25,
            "lines.linewidth": 2.5,
        })

        # ---------------- plots ----------------
        # ROC
        fig1, ax1 = plt.subplots()
        ax1.plot([0, 1], [0, 1], linestyle="--", linewidth=1.5, alpha=0.8, color="tab:blue", label="Chance")
        ax1.plot(fpr_grid, tpr_mean, color="tab:orange", label=f"Mean ROC (AUC={summary['roc_auc_mean']:.3f})")
         # draw bands conditionally
        if show_ci:
            ax1.fill_between(fpr_grid, tpr_lo, tpr_hi, alpha=0.2, label=f"{int(ci*100)}% CI")
        if show_std:
            ax1.fill_between(
                fpr_grid,
                np.clip(tpr_mean - tpr_sd, 0, 1),
                np.clip(tpr_mean + tpr_sd, 0, 1),
                alpha=0.12,
                label="±1 SD",
            )
        ax1.set_title(f"ROC Curve — runs={n_runs}")# + (f" {smoothing_note}" if smooth_lpf else ""))
        ax1.set_xlabel("False Positive Rate")
        ax1.set_ylabel("True Positive Rate")
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        ax1.grid(True, linestyle=":")
        ax1.legend(loc="lower right", frameon=True)
        roc_png = os.path.join(curve_dir, f"roc_mean_ci{'_smooth' if smooth_lpf else ''}.png")
        roc_svg = os.path.join(curve_dir, f"roc_mean_ci{'_smooth' if smooth_lpf else ''}.svg")
        fig1.tight_layout()
        fig1.savefig(roc_png, dpi=dpi)
        fig1.savefig(roc_svg)
        plt.close(fig1)

        # PR
        fig2, ax2 = plt.subplots()
        if pr_chance:
            if positive_class_ratio is None:
                raise ValueError("Positive class prevalence is required to plot PR chance line but not found.")
            ax2.plot([0, 1], [positive_class_ratio, positive_class_ratio],
                    linestyle="--", linewidth=1.5, alpha=0.8, color="tab:blue", label="Chance")
        ax2.plot(recall_grid, prec_mean, color="tab:blue", label=f"Mean PR (AP={summary['ap_mean']:.3f})")
        if show_ci:
            ax2.fill_between(recall_grid, prec_lo, prec_hi, alpha=0.2, label=f"{int(ci*100)}% CI")
        if show_std:
            ax2.fill_between(
                recall_grid,
                np.clip(prec_mean - prec_sd, 0, 1),
                np.clip(prec_mean + prec_sd, 0, 1),
                alpha=0.12,
                label="±1 SD",
            )
        ax2.set_title(f"Precision–Recall — runs={n_runs}")# + (f" {smoothing_note}" if smooth_lpf else ""))
        ax2.set_xlabel("Recall")
        ax2.set_ylabel("Precision")
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        ax2.grid(True, linestyle=":")
        ax2.legend(loc="lower left", frameon=True)
        pr_png = os.path.join(curve_dir, f"pr_mean_ci{suffix}.png")
        pr_svg = os.path.join(curve_dir, f"pr_mean_ci{suffix}.svg")
        fig2.tight_layout()
        fig2.savefig(pr_png, dpi=dpi)
        fig2.savefig(pr_svg)
        plt.close(fig2)

        if getattr(self, "verbose", False):
            self.printer(
                "[Curves] Aggregated plots saved:\n"
                f"  ROC: {roc_png}\n"
                f"  PR : {pr_png}\n"
                f"  (SVGs also saved)\n"
                f"  Summary: runs={n_runs}, AUC mean±std={summary['roc_auc_mean']:.4f}±{summary['roc_auc_std']:.4f}, "
                f"AP mean±std={summary['ap_mean']:.4f}±{summary['ap_std']:.4f} "
                f"{'(smoothed ' + filter_type + ')' if smooth_lpf else ''}\n"
            )

        return {
            "summary": summary,
            "roc_png": roc_png,
            "pr_png": pr_png,
            "accumulator": agg_path,
            "out_dir": curve_dir,
        }




        
    def collect_performance_metrics(
        self,
        model_pipeline,
        X,
        y,
        class_idx: int = 1,
        threshold: float = 0.5,
        experiment_dir: str = "results_unknown_date",
        run_name: str = None,
    ):
        """
        Compute a comprehensive set of binary classification metrics for a single run,
        append to an on-disk accumulator (.npz), and update a per-run CSV.

        Notes
        -----
        - Positive class is assumed to be label 1 in y. If different, remap before calling.
        - For binary models, uses column 1 of predict_proba or the appropriate column
        of decision_function; falls back to predicted labels if necessary.
        - 'threshold' applies to the score vector to derive predicted labels.
        """
        import os
        import numpy as np
        import pandas as pd
        from sklearn.metrics import (
            accuracy_score,
            precision_score,
            recall_score,
            f1_score,
            balanced_accuracy_score,
            matthews_corrcoef,
            roc_auc_score,
            average_precision_score,
            confusion_matrix,
            log_loss,
            brier_score_loss,
            precision_recall_curve,
            auc as sk_auc,
        )
        from sklearn.pipeline import Pipeline

        # --- dirs / files ---
        perf_dir = os.path.join(experiment_dir, "table", "performance")
        os.makedirs(perf_dir, exist_ok=True)
        agg_path = os.path.join(perf_dir, "performance_accumulator.npz")
        runs_csv = os.path.join(perf_dir, "per_run_performance.csv")

        # --- unpack estimator (for clarity only) ---
        estimator = model_pipeline
        if isinstance(model_pipeline, Pipeline) and model_pipeline.steps:
            estimator = model_pipeline.steps[-1][1]

        # --- decide which score vector to use (aligned with your curve util) ---
        scores = None
        if hasattr(model_pipeline, "predict_proba"):
            proba = model_pipeline.predict_proba(X)
            if proba.ndim == 2 and proba.shape[1] == 2:
                scores = proba[:, 1]
            elif proba.ndim == 2 and proba.shape[1] > 2:
                scores = proba[:, class_idx]
            else:
                scores = np.ravel(proba)
        if scores is None and hasattr(model_pipeline, "decision_function"):
            df = model_pipeline.decision_function(X)
            if df.ndim == 2:
                if df.shape[1] == 1:
                    scores = np.ravel(df)
                else:
                    use_idx = 1 if df.shape[1] == 2 else class_idx
                    scores = df[:, use_idx]
            else:
                scores = np.ravel(df)
        if scores is None:
            scores = np.ravel(model_pipeline.predict(X))  # fallback

        # --- predicted labels via threshold ---
        y = np.asarray(y).astype(int)
        y_pred = (scores >= threshold).astype(int)

        # --- confusion matrix components (pos_label = 1) ---
        tn, fp, fn, tp = confusion_matrix(y, y_pred, labels=[0, 1]).ravel()

        # --- core metrics ---
        has_prob_like = np.all(np.isfinite(scores))

        metrics = {}
        metrics["ACC"] = accuracy_score(y, y_pred)
        metrics["PPV"] = precision_score(y, y_pred, pos_label=1, zero_division=0)           # precision
        metrics["TPR"] = recall_score(y, y_pred, pos_label=1, zero_division=0)              # sensitivity/recall
        metrics["TNR"] = tn / (tn + fp) if (tn + fp) > 0 else 0.0                           # specificity
        metrics["FPR"] = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        metrics["FNR"] = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        metrics["F1"] = f1_score(y, y_pred, pos_label=1, zero_division=0)
        metrics["F1_Class1"] = f1_score(y, y_pred, pos_label=1, zero_division=0)
        metrics["F1_Class0"] = f1_score(y, y_pred, pos_label=0, zero_division=0)
        metrics["F1_Macro"] = f1_score(y, y_pred, average="macro", zero_division=0)
        metrics["F1_Weighted"] = f1_score(y, y_pred, average="weighted", zero_division=0)
        metrics["Balanced_Accuracy"] = balanced_accuracy_score(y, y_pred)
        metrics["MCC"] = matthews_corrcoef(y, y_pred) if (tp + tn + fp + fn) > 0 else 0.0
        metrics["NPV"] = tn / (tn + fn) if (tn + fn) > 0 else 0.0
        metrics["Prevalence"] = float(np.mean(y == 1))

        # --- area & score-based metrics ---
        # ROC AUC & AP
        try:
            metrics["ROC_AUC"] = roc_auc_score(y, scores)
        except Exception:
            metrics["ROC_AUC"] = np.nan
        try:
            metrics["Average_Precision"] = average_precision_score(y, scores)  # AP
        except Exception:
            metrics["Average_Precision"] = np.nan
        # PR_AUC by trapezoidal integration of PR curve
        try:
            precision, recall, _ = precision_recall_curve(y, scores, pos_label=1)
            metrics["PR_AUC"] = sk_auc(recall, precision)
        except Exception:
            metrics["PR_AUC"] = np.nan

        # Brier score & log loss (probability-like scores)
        if has_prob_like:
            try:
                from numpy import clip
                metrics["Brier_Score"] = brier_score_loss(y, clip(scores, 0, 1))
            except Exception:
                metrics["Brier_Score"] = np.nan
            try:
                p1 = np.clip(scores, 1e-7, 1 - 1e-7)
                p0 = 1.0 - p1
                proba_2col = np.vstack([p0, p1]).T
                metrics["Log_Loss"] = log_loss(y, proba_2col, labels=[0, 1])
            except Exception:
                metrics["Log_Loss"] = np.nan
        else:
            metrics["Brier_Score"] = np.nan
            metrics["Log_Loss"] = np.nan

        # counts / support
        metrics["TP"] = float(tp)
        metrics["TN"] = float(tn)
        metrics["FP"] = float(fp)
        metrics["FN"] = float(fn)
        metrics["N"] = float(y.size)
        metrics["Threshold"] = float(threshold)

        # --- persist in fixed order ---
        metric_names = [
            "ACC", "PPV", "TPR", "TNR", "FPR", "FNR",
            "F1", "F1_Class1", "F1_Class0", "F1_Macro", "F1_Weighted",
            "Balanced_Accuracy", "MCC",
            "ROC_AUC", "Average_Precision", "PR_AUC", "Brier_Score", "Log_Loss",
            "NPV", "Prevalence",
            "TP", "TN", "FP", "FN", "N", "Threshold",
        ]
        metric_values = np.array([metrics[m] for m in metric_names], dtype=float)

        this_name = run_name if run_name is not None else "run_0"
        if os.path.exists(agg_path):
            data = np.load(agg_path, allow_pickle=False)
            prev_names = data["metric_names"].astype("U256").tolist()
            if prev_names != metric_names:
                # metric set changed → start a new accumulator to avoid mismatches
                perf_stack = metric_values[None, :]
                run_names = np.array([this_name], dtype="U256")
                metric_names_arr = np.array(metric_names, dtype="U256")
            else:
                perf_stack = np.vstack([data["perf_stack"], metric_values[None, :]])
                prev_run_names = data["run_names"].astype("U256")
                if run_name is None:
                    this_name = f"run_{prev_run_names.size}"
                run_names = np.append(prev_run_names, np.array([this_name], dtype="U256"))
                metric_names_arr = data["metric_names"]
        else:
            perf_stack = metric_values[None, :]
            run_names = np.array([this_name], dtype="U256")
            metric_names_arr = np.array(metric_names, dtype="U256")

        np.savez_compressed(
            agg_path,
            perf_stack=perf_stack,      # shape (R, M)
            run_names=run_names,        # shape (R,)
            metric_names=metric_names_arr,  # shape (M,)
        )

        # per-run CSV (append)
        per_run_row = pd.DataFrame([{**{"run_name": this_name}, **{k: metrics[k] for k in metric_names}}])
        if os.path.exists(runs_csv):
            per_run_row.to_csv(runs_csv, mode="a", header=False, index=False)
        else:
            per_run_row.to_csv(runs_csv, index=False)

        # optional console printout in your style
        if getattr(self, "verbose", False):
            self.printer(
                "[Evaluator] Accuracy: {:.5f}\n"
                "[Evaluator] AUC: {:.5f}\n"
                "[Evaluator] F1 Score: {:.5f}\n"
                "F1 Score (Responder - Class 1): {:.5f}\n"
                "F1 Score (Non-Responder - Class 0): {:.5f}\n"
                "[Evaluator] MCC: {:.5f}\n"
                "[Evaluator] Sensitivity (Recall): {:.5f}\n"
                "[Evaluator] Specificity: {:.5f}\n"
                "[Evaluator] Balanced Accuracy: {:.5f}\n"
                "(accumulator: {})\n(per-run CSV: {})\n".format(
                    metrics["ACC"],
                    metrics["ROC_AUC"] if np.isfinite(metrics["ROC_AUC"]) else float("nan"),
                    metrics["F1"],
                    metrics["F1_Class1"],
                    metrics["F1_Class0"],
                    metrics["MCC"],
                    metrics["TPR"],
                    metrics["TNR"],
                    metrics["Balanced_Accuracy"],
                    agg_path,
                    runs_csv,
                )
            )

        return {
            "run_name": this_name,
            "metrics": metrics,
            "metric_names": metric_names,
            "accumulator": agg_path,
            "runs_csv": runs_csv,
            "out_dir": perf_dir,
        }


    def summarize_performance_metrics(
        self,
        experiment_dir: str = "results_unknown_date",
        ci: float = 0.95,
        save_csv: bool = True,
    ):
        """
        Load the performance accumulator and compute summary statistics across runs:
        mean, std, standard error, 95% CI, median, IQR, min, max for each metric.
        Saves CSVs into <experiment_dir>/table/performance/.
        """
        import os
        import numpy as np
        import pandas as pd
        from math import sqrt
        from scipy.stats import norm

        perf_dir = os.path.join(experiment_dir, "table", "performance")
        agg_path = os.path.join(perf_dir, "performance_accumulator.npz")
        os.makedirs(perf_dir, exist_ok=True)

        if not os.path.exists(agg_path):
            raise FileNotFoundError(
                f"No performance accumulator found at {agg_path}. "
                "Run `collect_performance_metrics` in your iterations first."
            )

        data = np.load(agg_path, allow_pickle=False)
        perf_stack = data["perf_stack"]       # (R, M)
        run_names = data["run_names"]         # (R,)
        metric_names = data["metric_names"]   # (M,)
        n_runs, n_metrics = perf_stack.shape
        z = norm.ppf(0.5 + ci / 2.0)

        # stats per metric (over runs)
        means = np.nanmean(perf_stack, axis=0)
        sds = np.nanstd(perf_stack, axis=0, ddof=1) if n_runs > 1 else np.zeros(n_metrics)
        ses = sds / sqrt(n_runs) if n_runs > 1 else np.zeros(n_metrics)
        ci_lo = means - z * ses
        ci_hi = means + z * ses

        medians = np.nanmedian(perf_stack, axis=0)
        q1 = np.nanpercentile(perf_stack, 25, axis=0)
        q3 = np.nanpercentile(perf_stack, 75, axis=0)
        iqrs = q3 - q1
        mins = np.nanmin(perf_stack, axis=0)
        maxs = np.nanmax(perf_stack, axis=0)

        summary_df = pd.DataFrame({
            "metric": metric_names,
            "n_runs": int(n_runs),
            "mean": means,
            "std": sds,
            "se": ses,
            f"ci_{int(ci*100)}_lo": ci_lo,
            f"ci_{int(ci*100)}_hi": ci_hi,
            "median": medians,
            "q1": q1,
            "q3": q3,
            "iqr": iqrs,
            "min": mins,
            "max": maxs,
        })

        # per-run table (kept in sync with accumulator)
        per_run_df = pd.DataFrame(perf_stack, columns=metric_names)
        per_run_df.insert(0, "run_name", run_names)

        if save_csv:
            summary_path = os.path.join(perf_dir, "performance_summary_stats.csv")
            runs_path = os.path.join(perf_dir, "per_run_performance.csv")
            summary_df.to_csv(summary_path, index=False)
            per_run_df.to_csv(runs_path, index=False)

        if getattr(self, "verbose", False):
            self.printer(
                "[Evaluator] Summary saved:\n"
                f"  {os.path.join(perf_dir, 'performance_summary_stats.csv')}\n"
                f"  {os.path.join(perf_dir, 'per_run_performance.csv')}\n"
                f"  runs={n_runs}, metrics={len(metric_names)}\n"
            )

        return {
            "n_runs": int(n_runs),
            "metrics": [str(m) for m in metric_names],
            "summary_csv": os.path.join(perf_dir, "performance_summary_stats.csv") if save_csv else None,
            "per_run_csv": os.path.join(perf_dir, "per_run_performance.csv") if save_csv else None,
            "accumulator": agg_path,
            "out_dir": perf_dir,
        }

if __name__ == "__main__":
    model_evaluator = ModelEvaluator(verbose=True)
    experiment_dir="/home/junfu.cheng/SMILE/github/j_map_2025_8_16/results_2025_09_14_13_27_47"
    model_evaluator.plot_aggregated_curves(experiment_dir=experiment_dir, smooth_lpf=True)


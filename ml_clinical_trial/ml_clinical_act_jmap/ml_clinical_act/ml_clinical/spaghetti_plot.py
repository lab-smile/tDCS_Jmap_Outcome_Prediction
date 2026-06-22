#
#  spaghetti_plot.py
#  smile-rf1-j-map-prediction
'''
This code module is designed to extract and visualize state anxiety scores from your datasets. To use it, first ensure your datasets (typically named `dataset_in_use` and `act_data`) are loaded into your workspace. Specify the variable of interest (by default `'stai_state_score_i'`), then call the `extract_score_data()` function to retrieve the baseline and post-intervention scores along with the responder status. Next, instantiate the `STAIScorePlotter` class using the extracted data, and finally call the `plot()` method to generate a multi-panel plot that illustrates the distribution of scores across different groups and conditions. The plot is both saved as a PDF file and displayed, providing a comprehensive visualization of your data.
'''
#  Created by Cheng, Junfu on 4/3/25.
#
import os
import numpy as np
import pandas as pd
from scipy.stats import shapiro, levene, ttest_ind, mannwhitneyu
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.stats import gaussian_kde

from enum import Enum

class StatsTimePoint(Enum):
    PRE = 1
    POST = 2
    TRAINING = 3
    TESTING = 4

# def extract_score_data(dataset_in_use, act_data, variable_name='stai_state_score_i'):
#     """
#     Extracts the baseline and post-intervention scores and responder info.
#     Returns:
#         tp1_df: DataFrame with baseline scores
#         tp2_df: DataFrame with post-intervention scores
#         responder: DataFrame with responder info (mapped to labels)
#     """
#     # Extract baseline and post-intervention scores
#     tp1_df = dataset_in_use[[variable_name]]
#     tp2_df = act_data.tp2[[variable_name]]
    
#     # Extract responder info and map to labels
#     responder = dataset_in_use[['target']].replace({0: 'Non-responder', 1: 'Responder'})
    
#     return tp1_df, tp2_df, responder

def extract_score_data_from_ids(df, subject_ids, results, targets):
    """
    Extracts baseline and post-intervention scores and responder info 
    using subject IDs and specified target columns.
    
    Args:
        df (pd.DataFrame): Full dataframe containing tp1, tp2, and target columns.
        subject_ids (list): List of subject indices (rows) to include.
        targets (list): List of two column names [tp1_column, tp2_column].
    
    Returns:
        tp1_df (pd.DataFrame): DataFrame with baseline scores (tp1).
        tp2_df (pd.DataFrame): DataFrame with post-intervention scores (tp2).
        responder (pd.DataFrame): DataFrame with responder info (mapped to labels).
    """
    # Subset dataframe by subject IDs
    df_sub = df.loc[subject_ids]
    
    # Extract tp1 and tp2 scores
    tp1_df = df_sub[[targets[0]]]
    tp2_df = df_sub[[targets[1]]]
    
    # Extract responder info and map to labels
    responder = results[['responder']].replace({0: 'Non-responder', 1: 'Responder'})
    
    return tp1_df, tp2_df, responder

def _descriptives(x: pd.Series) -> dict:
    """Compute n, mean, median, SD (sample), IQR for a 1D array-like."""
    x = pd.Series(x).dropna()
    if x.empty:
        return {"n": 0, "mean": np.nan, "median": np.nan, "sd": np.nan, "iqr": np.nan}
    return {
        "n": x.shape[0],
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "sd": float(np.std(x, ddof=1)),
        "iqr": float(np.percentile(x, 75) - np.percentile(x, 25)),
    }

def _hedges_g(x, y) -> float:
    """
    Small-sample corrected standardized mean difference.
    Cohen's d with Hedges' J correction (Hedges, 1981).
    """
    x = pd.Series(x).dropna()
    y = pd.Series(y).dropna()
    nx, ny = len(x), len(y)
    sx2 = np.var(x, ddof=1)
    sy2 = np.var(y, ddof=1)
    # Pooled SD (equal-variance form) — used for reporting effect size only;
    # test choice (t vs Welch) is still determined separately.
    sp = np.sqrt(((nx - 1) * sx2 + (ny - 1) * sy2) / (nx + ny - 2))
    if sp == 0:
        return np.nan
    d = (np.mean(x) - np.mean(y)) / sp
    J = 1 - (3 / (4 * (nx + ny) - 9))  # small-sample correction
    return float(J * d)

def _rank_biserial_from_U(U, n1, n2) -> float:
    """
    Rank-biserial correlation from Mann–Whitney U (Kerby, 2014).
    r_rb = 1 - 2U/(n1*n2)
    """
    if n1 == 0 or n2 == 0:
        return np.nan
    return float(1 - (2 * U) / (n1 * n2))

def summarize_and_compare(tp1_df, tp2_df, responder, group_labels=None):
    """
    Compute descriptive and inferential statistics for two-group comparisons.

    Evidence & rationale behind method selection (short notes):
    - Normality: Shapiro–Wilk is powerful for small-to-moderate n (Shapiro & Wilk, 1965).
    - Homogeneity: Levene’s test is robust to non-normality (Levene, 1960).
    - If both groups ~normal:
        • Equal variances → independent-samples t-test.
        • Unequal variances → Welch’s t-test (robust to heteroscedasticity and unequal n; Welch, 1947).
      Effect size: Hedges’ g (small-sample corrected standardized mean difference; Hedges, 1981).
    - If ≥1 group non-normal:
        • Mann–Whitney U (distribution-free, tests stochastic dominance; Mann & Whitney, 1947).
      Effect size: rank-biserial correlation r (intuitive probability-of-superiority link; Kerby, 2014).

    Args:
        tp1_df (pd.DataFrame): Baseline (tp1) scores, single column.
        tp2_df (pd.DataFrame): Post (tp2) scores, single column.
        responder (pd.DataFrame): Labels (e.g., 'Responder' / 'Non-responder').
        group_labels (pd.Series | np.array | list, optional): Alternative grouping
            (e.g., 'Active'/'Sham'). If None, uses `responder`.

    Returns:
        dict:
          {
            "descriptive": {
               "Baseline": {<group or 'All'>: {n, mean, median, sd, iqr}, ...},
               "Post":     {...},
               "Change":   {...}
            },
            "inferential": {
               "groups": [g1, g2],
               "test": "Independent samples t-test" | "Welch’s t-test" | "Mann–Whitney U test",
               "statistic": <t or U>,
               "p_value": <float>,
               "effect_size": ("Hedges’ g" | "Rank-biserial r", <value>),
               "assumptions": {
                   "Shapiro p (<group1>)": <float or nan>,
                   "Shapiro p (<group2>)": <float or nan>,
                   "Levene p": <float or nan>
               }
            }  # or {} if not exactly 2 groups are present
          }
    """
    # ---- Build analysis frame
    group_col = (pd.Series(group_labels).reset_index(drop=True)
                 if group_labels is not None
                 else pd.Series(responder.values.flatten()).reset_index(drop=True))

    df = pd.DataFrame({
        "Baseline": pd.Series(tp1_df.values.flatten(), dtype=float).reset_index(drop=True),
        "Post": pd.Series(tp2_df.values.flatten(), dtype=float).reset_index(drop=True),
        "Group": group_col.astype("category")
    })
    df["Change"] = df["Post"] - df["Baseline"]

    results = {"descriptive": {}, "inferential": {}}

    # ---- Descriptive statistics (per group + All)
    for measure in ["Baseline", "Post", "Change"]:
        stats_table = {}
        # Per group
        for g in df["Group"].cat.categories:
            stats_table[str(g)] = _descriptives(df.loc[df["Group"] == g, measure])
        # All
        stats_table["All"] = _descriptives(df[measure])
        results["descriptive"][measure] = stats_table

    # ---- Inferential statistics (Post only, exactly two groups)
    cats = list(df["Group"].cat.categories)
    if len(cats) == 2:
        g1, g2 = cats[0], cats[1]
        x = df.loc[df["Group"] == g1, "Post"].dropna()
        y = df.loc[df["Group"] == g2, "Post"].dropna()

        # Guard for tiny samples
        shapiro_x = shapiro(x) if len(x) >= 3 else (np.nan, np.nan)
        shapiro_y = shapiro(y) if len(y) >= 3 else (np.nan, np.nan)
        lev = levene(x, y) if (len(x) > 1 and len(y) > 1) else (np.nan, np.nan)

        # Decision logic
        normal_x = (not np.isnan(shapiro_x[1])) and (shapiro_x[1] > 0.05)
        normal_y = (not np.isnan(shapiro_y[1])) and (shapiro_y[1] > 0.05)
        equal_var = (not np.isnan(lev[1])) and (lev[1] > 0.05)

        if normal_x and normal_y:
            if equal_var:
                test_name = "Independent samples t-test"
                stat, p = ttest_ind(x, y, equal_var=True)
            else:
                test_name = "Welch’s t-test"
                stat, p = ttest_ind(x, y, equal_var=False)
            effect = ("Hedges’ g", _hedges_g(x, y))
        else:
            test_name = "Mann–Whitney U test"
            stat, p = mannwhitneyu(x, y, alternative="two-sided")
            effect = ("Rank-biserial r", _rank_biserial_from_U(stat, len(x), len(y)))

        results["inferential"] = {
            "groups": [str(g1), str(g2)],
            "test": test_name,
            "statistic": float(stat) if stat is not None else np.nan,
            "p_value": float(p) if p is not None else np.nan,
            "effect_size": (effect[0], float(effect[1]) if effect[1] is not None else np.nan),
            "assumptions": {
                f"Shapiro p ({g1})": float(shapiro_x[1]) if not np.isnan(shapiro_x[1]) else np.nan,
                f"Shapiro p ({g2})": float(shapiro_y[1]) if not np.isnan(shapiro_y[1]) else np.nan,
                "Levene p": float(lev[1]) if not np.isnan(lev[1]) else np.nan
            }
        }

    return results

def save_stats_to_csv(stats, save_stats_time_point, out_dir):
    """
    Save descriptive and inferential statistics into CSV files:
      out_dir/table/stats_tables/descriptive_stats.csv
      out_dir/table/stats_tables/inferential_stats.csv

    Descriptive CSV is in long-ish wide format (each row = one group within a measure).
    """
    table_path = os.path.join(out_dir, "table", "stats_tables")
    os.makedirs(table_path, exist_ok=True)
    if save_stats_time_point == StatsTimePoint.PRE:
        table_path = os.path.join(out_dir, "table", "stats_tables_pre")
        os.makedirs(table_path, exist_ok=True)
    if save_stats_time_point == StatsTimePoint.POST:
        table_path = os.path.join(out_dir, "table", "stats_tables_post")
        os.makedirs(table_path, exist_ok=True)
    if save_stats_time_point == StatsTimePoint.TRAINING:
        table_path = os.path.join(out_dir, "table", "stats_tables_training")
        os.makedirs(table_path, exist_ok=True)
    if save_stats_time_point == StatsTimePoint.TESTING:
        table_path = os.path.join(out_dir, "table", "stats_tables_testing")
        os.makedirs(table_path, exist_ok=True)

    # ---- Descriptive
    desc_frames = []
    for measure, table in stats["descriptive"].items():
        dframe = pd.DataFrame(table).T  # groups as rows
        dframe.insert(0, "Measure", measure)
        dframe = dframe.reset_index().rename(columns={"index": "Group"})
        # Ensure consistent column order
        cols = ["Measure", "Group", "n", "mean", "median", "sd", "iqr"]
        dframe = dframe[[c for c in cols if c in dframe.columns]]
        desc_frames.append(dframe)

    if len(desc_frames):
        descriptive_df = pd.concat(desc_frames, ignore_index=True)
        descriptive_df.to_csv(os.path.join(table_path, "descriptive_stats.csv"), index=False)

    # ---- Inferential
    inferential = stats.get("inferential", {})
    if inferential:
        effect_name, effect_value = inferential.get("effect_size", (None, np.nan))
        infer_df = pd.DataFrame([{
            "Group_1": inferential.get("groups", ["", ""])[0] if "groups" in inferential else "",
            "Group_2": inferential.get("groups", ["", ""])[1] if "groups" in inferential else "",
            "Test": inferential.get("test", ""),
            "Statistic": inferential.get("statistic", np.nan),
            "p_value": inferential.get("p_value", np.nan),
            "Effect_size_type": effect_name,
            "Effect_size_value": effect_value,
            **inferential.get("assumptions", {})
        }])
        infer_df.to_csv(os.path.join(table_path, "inferential_stats.csv"), index=False)

    print(f"[STAIScorePlotter] Stats saved to: {table_path}")

def stats_aggregate(tp1_df, tp2_df, responder, labeler, stats_time_point):
    # calculate descriptive statistics for the target feature at each time point, stratified by responder status
    # and inferential statistics to compare the target feature between responders and non-responders at each time point
    labeler.tp1_df, labeler.tp2_df, labeler.responder = tp1_df, tp2_df, responder
    stats = summarize_and_compare(tp1_df, tp2_df, responder)
    save_stats_to_csv(stats, stats_time_point, labeler.experiment_dir)
    plotter = STAIScorePlotter(tp1_df, tp2_df, responder, variable_name=labeler.variable_name)
    plotter.plot_severe(stats_time_point, labeler.experiment_dir )
    print(f"[DataLabeler] Spaghetti plot saved for severe subjects with above median decrease criteria.")


class STAIScorePlotter:
    def __init__(self, tp1_df, tp2_df, responder, variable_name='stai_state_score'):
        self.tp1_df = tp1_df
        self.tp2_df = tp2_df
        self.responder = responder
        self.variable_name = variable_name
        self.jitter_lookup = {}  # to store exact x positions for each subject's timepoints
        self.jitter_strength = 0.01  # scatter jitter strength
        self.prepare_data()
    
    def prepare_data(self):
        # Create base DataFrame using pre-extracted tp1_df, tp2_df, and responder
        data = pd.DataFrame({
            'ID': np.arange(len(self.tp1_df)),  # Unique ID for each participant
            'Baseline': self.tp1_df[self.variable_name + '_tp1'].values,
            'Post': self.tp2_df[self.variable_name + '_tp2'].values,
            'Responder': self.responder['responder'].values
        })
    
        # Derive condition based on baseline score
        data['Condition'] = np.where(
            data['Baseline'] < 39,
            'Mild',
            'ModerateSevere'
        )
    
        # Derive group label from responder status
        data['Group'] = data['Responder'].map({
            'Responder': 'Active',
            'Non-responder': 'Sham'
        })
    
        # Convert the wide-format DataFrame into long-format
        self.df_all = pd.melt(
            data,
            id_vars=['ID', 'Group', 'Condition'],
            value_vars=['Baseline', 'Post'],
            var_name='Time',
            value_name='Score'
        )
    
        self.df_all = self.df_all.sort_values(by=['ID', 'Time']).reset_index(drop=True)
    
        # Prepare datasets for different conditions
        self.df_all_conditions = self.df_all.copy()
        self.df_mild = self.df_all[self.df_all["Condition"] == "Mild"]
        self.df_modsev = self.df_all[self.df_all["Condition"] == "ModerateSevere"]
    
    @staticmethod
    def darken_color(color, amount=0.7):
        c = mcolors.to_rgb(color)
        return tuple([max(min(x * amount, 1.0), 0.0) for x in c])
    
    def plot(self, experiment_dir=None):
        # Define subplot setup for the final multi-panel plot
        fig, axes = plt.subplots(1, 3, figsize=(21, 7))  # 1 row, 3 columns
    
        panel_data = [
            ("A", "All", self.df_all_conditions),
            ("B", "Mild", self.df_mild),
            ("C", "ModerateSevere", self.df_modsev)
        ]
    
        groups = ["Sham", "Active"]
        group_colors = {"Sham": "skyblue", "Active": "lightcoral"}
        group_offsets = {"Sham": 0.12, "Active": 0.12}
        group_offsets_box_plot = {"Sham": 0.06, "Active": 0.10}
        group_offsets_scattor = {"Sham": 0.0, "Active": 0.0}
        jitter_strength = self.jitter_strength
    
        x_base = [0, 1]
        times = ["Baseline", "Post"]
        times_label = ["Baseline", "Post-intervention"]
        cond_label_i = [
            'All eligible participants',
            'Minimal to mild baseline state anxiety',
            'Moderate to severe baseline state anxiety'
        ]
    
        fontsize = 20
    
        for iteration, (ax, (panel, cond_label, df_plot)) in enumerate(zip(axes, panel_data)):
            # Plot half violins for each group and time
            for group in groups:
                for idx, t in enumerate(times):
                    vals = df_plot[(df_plot["Group"] == group) & (df_plot["Time"] == t)]["Score"].values
                    if len(vals) == 0:
                        continue
                    kde = gaussian_kde(vals)
                    y = np.linspace(vals.min() - 5, vals.max() + 5, 200)
                    v = kde(y)
                    v = v / v.max() * 0.2
                    offset = group_offsets[group]
                    direction = -1 if t == "Baseline" else 1
                    x_center = idx
                    ax.fill_betweenx(y, x_center + direction * offset,
                                     x_center + direction * (offset + v * np.sign(offset)),
                                     alpha=0.6, color=group_colors[group], linewidth=0)
    
            # Add box plots for each group and time
            box_width = 0.025
            for idx, t in enumerate(times):
                for group in groups:
                    vals = df_plot[(df_plot["Group"] == group) & (df_plot["Time"] == t)]["Score"].values
                    if len(vals) == 0:
                        continue
                    q1, med, q3 = np.percentile(vals, [25, 50, 75])
                    iqr = q3 - q1
                    lower_whisker = np.max([q1 - 1.5 * iqr, vals.min()])
                    upper_whisker = np.min([q3 + 1.5 * iqr, vals.max()])
                    offset = group_offsets_box_plot[group]
                    direction = -1 if t == "Baseline" else 1
                    center = idx + direction * offset
                    ax.add_patch(plt.Rectangle((center - box_width/2, q1), box_width, q3 - q1,
                                               facecolor=group_colors[group],
                                               edgecolor=self.darken_color(group_colors[group]),
                                               linewidth=1.2, zorder=3))
                    ax.plot([center - box_width/2, center + box_width/2], [med, med],
                            color=self.darken_color(group_colors[group]), linewidth=2.0, zorder=4)
                    ax.plot([center, center], [q3, upper_whisker],
                            color=self.darken_color(group_colors[group]), linewidth=1.0, zorder=2)
                    ax.plot([center, center], [q1, lower_whisker],
                            color=self.darken_color(group_colors[group]), linewidth=1.0, zorder=2)
                    cap_width = box_width * 0.1
                    ax.plot([center - cap_width/2, center + cap_width/2], [upper_whisker]*2,
                            color=self.darken_color(group_colors[group]), linewidth=1.0)
                    ax.plot([center - cap_width/2, center + cap_width/2], [lower_whisker]*2,
                            color=self.darken_color(group_colors[group]), linewidth=1.0)
    
            # Scatter points with horizontal jitter; store positions for line plotting
            for group in groups:
                for idx, t in enumerate(times):
                    group_data = df_plot[(df_plot["Group"] == group) & (df_plot["Time"] == t)]
                    subject_ids = group_data["ID"].values
                    x_center = idx + group_offsets_scattor[group]
                    jittered_x = np.random.normal(loc=x_center, scale=jitter_strength, size=len(group_data))
    
                    for sid, x in zip(subject_ids, jittered_x):
                        self.jitter_lookup[(sid, t)] = x
    
                    # Plot scatter points
                    marker = 'o'
                    color = group_colors[group]
                    ax.scatter(
                        jittered_x,
                        group_data["Score"],
                        color=color,
                        marker=marker,
                        s=40,
                        alpha=0.5,
                        edgecolors=self.darken_color(color),
                        linewidths=0.5,
                        zorder=5
                    )
    
            # Plot Mean ± Standard Deviation for each group
            for group in groups:
                means = []
                stds = []
                for t in times:
                    scores = df_plot[(df_plot["Group"] == group) & (df_plot["Time"] == t)]["Score"]
                    means.append(scores.mean())
                    stds.append(scores.std())
                x_vals = [x_base[times.index(t)] for t in times]
                ax.errorbar(
                    x_vals,
                    means,
                    yerr=stds,
                    color=self.darken_color(group_colors[group], amount=0.9),
                    fmt='-o',
                    capsize=8,
                    capthick=4,
                    linewidth=4,
                    markersize=12,
                    label=f"{group} Mean ± SD",
                    zorder=6
                )
    
            # Draw lines connecting scores for each subject using jittered positions
            for subject_id in df_plot["ID"].unique():
                d = df_plot[df_plot["ID"] == subject_id].sort_values("Time")
                group = d["Group"].iloc[0]
                y_vals = d["Score"].values
                x_vals = [self.jitter_lookup[(subject_id, t)] for t in d["Time"]]
                ax.plot(
                    x_vals,
                    y_vals,
                    color=group_colors[group],
                    alpha=0.3,
                    linewidth=0.8,
                    zorder=1
                )
    
            ax.set_xticks(x_base)
            ax.set_xticklabels(times_label, fontsize=fontsize)   # X-axis tick labels
            ax.tick_params(axis='y', labelsize=fontsize)         # Y-axis tick labels
            ax.set_xlim(-0.5, 1.5)
            ax.set_ylim(10, 60)
            ax.set_title(f"{panel}) {cond_label_i[iteration]}", fontsize=fontsize, pad=20)
            ax.set_xlabel("Time", fontsize=fontsize, fontweight='bold')
            ax.set_ylabel("STAI State Scores", fontsize=fontsize, fontweight='bold')
            ax.grid(True, axis='y', linestyle='--', alpha=0.3)
    
        # Global legend at the bottom center of the figure
        fig.legend(
            handles=[
                plt.Line2D([0], [0], color='skyblue', lw=6, label='Non-responder'),
                plt.Line2D([0], [0], color='lightcoral', lw=6, label='Responder')
            ],
            title="Response status",
            loc="lower center",
            ncol=2,
            bbox_to_anchor=(0.5, -0.12),
            frameon=False,
            fontsize=20,
            title_fontsize=20
        )
    
        plt.tight_layout()
        if experiment_dir:
            pdf_filename = 'stai_state_score_responder.pdf'
            png_filename = 'stai_state_score_responder.png'
            svg_filename = 'stai_state_score_responder.svg'
            save_dir = os.path.join(experiment_dir, 'figure', 'spaghetti_plot')
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, pdf_filename), format="pdf", bbox_inches="tight")
            plt.savefig(os.path.join(save_dir, png_filename), format="png", bbox_inches="tight", dpi=600)
            plt.savefig(os.path.join(save_dir, svg_filename), format="svg", bbox_inches="tight")
            print(f"[STAIScorePlotter] Spaghetti plot saved to {save_dir} as:")
            print(pdf_filename)
            print(png_filename)
            print(svg_filename)
        else:
            plt.show()

    def plot_severe(self, save_stats_time_point, experiment_dir):
        import os
        import numpy as np
        import matplotlib.pyplot as plt
        from scipy.stats import gaussian_kde

        # Reset jitter lookup so lines connect correctly for this plot only
        self.jitter_lookup = {}

        # Single-panel figure (same height as before)
        fig, ax = plt.subplots(1, 1, figsize=(7, 7))

        # Data and labels (only ModerateSevere)
        df_plot = self.df_modsev
        cond_label_i = "Moderate to severe baseline state anxiety"

        groups = ["Sham", "Active"]
        group_colors = {"Sham": "skyblue", "Active": "lightcoral"}
        group_offsets = {"Sham": 0.12, "Active": 0.12}
        group_offsets_box_plot = {"Sham": 0.06, "Active": 0.10}
        group_offsets_scattor = {"Sham": 0.0, "Active": 0.0}
        jitter_strength = self.jitter_strength

        x_base = [0, 1]
        times = ["Baseline", "Post"]
        times_label = ["Baseline", "Post-intervention"]

        fontsize = 20

        # Half violins
        for group in groups:
            for idx, t in enumerate(times):
                vals = df_plot[(df_plot["Group"] == group) & (df_plot["Time"] == t)]["Score"].values
                if len(vals) == 0:
                    continue
                kde = gaussian_kde(vals)
                y = np.linspace(vals.min() - 5, vals.max() + 5, 200)
                v = kde(y)
                v = v / v.max() * 0.2
                offset = group_offsets[group]
                direction = -1 if t == "Baseline" else 1
                x_center = idx
                ax.fill_betweenx(
                    y,
                    x_center + direction * offset,
                    x_center + direction * (offset + v * np.sign(offset)),
                    alpha=0.6,
                    color=group_colors[group],
                    linewidth=0
                )

        # Box plots
        box_width = 0.025
        for idx, t in enumerate(times):
            for group in groups:
                vals = df_plot[(df_plot["Group"] == group) & (df_plot["Time"] == t)]["Score"].values
                if len(vals) == 0:
                    continue
                q1, med, q3 = np.percentile(vals, [25, 50, 75])
                iqr = q3 - q1
                lower_whisker = np.max([q1 - 1.5 * iqr, vals.min()])
                upper_whisker = np.min([q3 + 1.5 * iqr, vals.max()])
                offset = group_offsets_box_plot[group]
                direction = -1 if t == "Baseline" else 1
                center = idx + direction * offset
                ax.add_patch(plt.Rectangle(
                    (center - box_width/2, q1),
                    box_width,
                    q3 - q1,
                    facecolor=group_colors[group],
                    edgecolor=self.darken_color(group_colors[group]),
                    linewidth=1.2,
                    zorder=3
                ))
                ax.plot([center - box_width/2, center + box_width/2], [med, med],
                        color=self.darken_color(group_colors[group]), linewidth=2.0, zorder=4)
                ax.plot([center, center], [q3, upper_whisker],
                        color=self.darken_color(group_colors[group]), linewidth=1.0, zorder=2)
                ax.plot([center, center], [q1, lower_whisker],
                        color=self.darken_color(group_colors[group]), linewidth=1.0, zorder=2)
                cap_width = box_width * 0.1
                ax.plot([center - cap_width/2, center + cap_width/2], [upper_whisker]*2,
                        color=self.darken_color(group_colors[group]), linewidth=1.0)
                ax.plot([center - cap_width/2, center + cap_width/2], [lower_whisker]*2,
                        color=self.darken_color(group_colors[group]), linewidth=1.0)

        # Scatter with jitter + store jitter for connecting lines
        for group in groups:
            for idx, t in enumerate(times):
                group_data = df_plot[(df_plot["Group"] == group) & (df_plot["Time"] == t)]
                subject_ids = group_data["ID"].values
                x_center = idx + group_offsets_scattor[group]
                jittered_x = np.random.normal(loc=x_center, scale=jitter_strength, size=len(group_data))

                for sid, x in zip(subject_ids, jittered_x):
                    self.jitter_lookup[(sid, t)] = x

                ax.scatter(
                    jittered_x,
                    group_data["Score"],
                    color=group_colors[group],
                    marker='o',
                    s=40,
                    alpha=0.5,
                    edgecolors=self.darken_color(group_colors[group]),
                    linewidths=0.5,
                    zorder=5
                )

        # Mean ± SD lines
        for group in groups:
            means, stds = [], []
            for t in times:
                scores = df_plot[(df_plot["Group"] == group) & (df_plot["Time"] == t)]["Score"]
                means.append(scores.mean())
                stds.append(scores.std())
            x_vals = [x_base[times.index(t)] for t in times]
            ax.errorbar(
                x_vals,
                means,
                yerr=stds,
                color=self.darken_color(group_colors[group], amount=0.9),
                fmt='-o',
                capsize=8,
                capthick=4,
                linewidth=4,
                markersize=12,
                label=f"{group} Mean ± SD",
                zorder=6
            )

        # Subject spaghetti lines
        for subject_id in df_plot["ID"].unique():
            d = df_plot[df_plot["ID"] == subject_id].sort_values("Time")
            group = d["Group"].iloc[0]
            y_vals = d["Score"].values
            x_vals = [self.jitter_lookup[(subject_id, t)] for t in d["Time"]]
            ax.plot(
                x_vals,
                y_vals,
                color=group_colors[group],
                alpha=0.3,
                linewidth=0.8,
                zorder=1
            )

        # Axes cosmetics
        ax.set_xticks(x_base)
        ax.set_xticklabels(times_label, fontsize=fontsize)   # X-axis tick labels
        ax.tick_params(axis='y', labelsize=fontsize)         # Y-axis tick labels
        ax.set_xlim(-0.5, 1.5)
        ax.set_ylim(10, 60)
        ax.set_title(f"{cond_label_i}", fontsize=fontsize, pad=20)
        ax.set_xlabel("Time", fontsize=fontsize, fontweight='bold')
        ax.set_ylabel("STAI State Scores", fontsize=fontsize, fontweight='bold')
        ax.grid(True, axis='y', linestyle='--', alpha=0.3)

        # Global legend (same labels/colors as your original)
        fig.legend(
            handles=[
                plt.Line2D([0], [0], color='skyblue', lw=6, label='Non-responder'),
                plt.Line2D([0], [0], color='lightcoral', lw=6, label='Responder')
            ],
            title="Response status",
            loc="lower center",
            ncol=2,
            bbox_to_anchor=(0.5, -0.12),
            frameon=False,
            fontsize=20,
            title_fontsize=20
        )

        plt.tight_layout()
        if experiment_dir:
            pdf_filename = 'stai_state_score_responder_severe.pdf'
            png_filename = 'stai_state_score_responder_severe.png'
            svg_filename = 'stai_state_score_responder_severe.svg'
            if save_stats_time_point == StatsTimePoint.PRE:
                save_dir = os.path.join(experiment_dir, 'figure', 'spaghetti_plot_pre')
            if save_stats_time_point == StatsTimePoint.POST:
                save_dir = os.path.join(experiment_dir, 'figure', 'spaghetti_plot_post')
            if save_stats_time_point == StatsTimePoint.TRAINING:
                save_dir = os.path.join(experiment_dir, 'figure', 'spaghetti_plot_training')
            if save_stats_time_point == StatsTimePoint.TESTING:
                save_dir = os.path.join(experiment_dir, 'figure', 'spaghetti_plot_testing')
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, pdf_filename), format="pdf", bbox_inches="tight")
            plt.savefig(os.path.join(save_dir, png_filename), format="png", dpi=600, bbox_inches="tight")
            plt.savefig(os.path.join(save_dir, svg_filename), format="svg", bbox_inches="tight")
            print(f"[STAIScorePlotter] Severe-only spaghetti plot saved to {save_dir} as:")
            print(pdf_filename)
            print(png_filename)
            print(svg_filename)
        else:
            plt.show()

        
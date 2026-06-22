import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def _sanitize_col(text: str) -> str:
    """
    Sanitize category values into safe column names.
    Examples:
      'Strongly agree' -> 'Strongly_agree'
      '0 = No'         -> '0_No'
    """
    s = str(text)
    s = re.sub(r"[=/,;:|]+", " ", s)
    s = re.sub(r"[^\w\s-]", "", s)
    s = re.sub(r"[\s-]+", "_", s).strip("_")
    return s or "category"

def characterize_dataset(df, metadata_df=None, prefix="train", categorical_threshold=10):
    """
    Characterize dataset:
      - Categorical: variables with < categorical_threshold unique (non-null) values
      - Numeric: everything else (attempt numeric coercion)

    Outputs:
      - Per-variable CSVs/plots
      - Global summary CSV with:
        ['variable','label','type','n_nonnull','mean','std','n_unique_categories', <category columns>]
        For each categorical variable, adds one column per category with counts.
    """

    # Directories
    base_dir = "./data_generation_log/generated_table/figure/"
    fig_dir = os.path.join(base_dir, prefix, "figures")
    table_dir = os.path.join(base_dir, prefix, "tables")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(table_dir, exist_ok=True)

    # Metadata mapping
    label_map = {}
    if metadata_df is not None and "variable" in metadata_df.columns:
        for _, r in metadata_df.iterrows():
            v = r.get("variable")
            if pd.isna(v):
                continue
            label_map[str(v)] = r.get("label", v)

    summary_rows = []

    for var in df.columns:
        series = df[var].dropna()
        label = label_map.get(var, var)

        if series.empty:
            summary_rows.append({
                "variable": var,
                "label": label,
                "type": "unknown",
                "n_nonnull": 0,
                "mean": np.nan,
                "std": np.nan,
                "n_unique_categories": np.nan
            })
            print(f"[info] {label} ({var}): no non-null values; skipping.")
            continue

        n_unique = int(series.nunique(dropna=True))
        is_categorical = n_unique < categorical_threshold

        if is_categorical:
            # Frequencies
            series_str = series.astype(str)
            freq = series_str.value_counts(dropna=True)

            # Save frequency table
            per_table_path = os.path.join(table_dir, f"{var}_freq.csv")
            freq.to_csv(per_table_path, header=["count"])
            print(f"[categorical] {label} ({var})")
            print(f"  distinct categories = {n_unique}")
            for cat, cnt in freq.items():
                print(f"    - {cat}: {cnt}")
            print(f"  saved table: {os.path.abspath(per_table_path)}")

            # Plot
            plt.figure()
            plt.bar(freq.index.astype(str), freq.values)
            plt.title(f"{label} ({var}) - {prefix}")
            plt.xlabel(var)
            plt.ylabel("Count")
            plt.xticks(rotation=45, ha="right")
            per_fig_path = os.path.join(fig_dir, f"{var}_bar.png")
            plt.tight_layout()
            plt.savefig(per_fig_path, bbox_inches="tight")
            plt.close()
            print(f"  saved figure: {os.path.abspath(per_fig_path)}")

            # Summary row
            row_dict = {
                "variable": var,
                "label": label,
                "type": "categorical",
                "n_nonnull": int(series_str.shape[0]),
                "mean": np.nan,
                "std": np.nan,
                "n_unique_categories": n_unique
            }
            for cat, cnt in freq.items():
                safe_cat = _sanitize_col(cat)
                row_dict[safe_cat] = int(cnt)
            summary_rows.append(row_dict)

        else:
            # Numeric
            series_num = pd.to_numeric(series, errors="coerce").dropna()
            if series_num.empty:
                print(f"[warn] {label} ({var}): not numeric after coercion.")
                summary_rows.append({
                    "variable": var,
                    "label": label,
                    "type": "numeric",
                    "n_nonnull": int(series.shape[0]),
                    "mean": np.nan,
                    "std": np.nan,
                    "n_unique_categories": np.nan
                })
            else:
                desc = series_num.describe()
                mean_val = float(series_num.mean())
                std_val = float(series_num.std(ddof=1))

                # Save per-variable stats
                per_table_path = os.path.join(table_dir, f"{var}_summary.csv")
                desc.to_csv(per_table_path)
                print(f"[numeric] {label} ({var})")
                print(f"  mean = {mean_val:.6g}, std = {std_val:.6g}")
                print(f"  saved table: {os.path.abspath(per_table_path)}")

                # Plot
                plt.figure()
                plt.hist(series_num, bins="auto", edgecolor="black")
                plt.title(f"{label} ({var}) - {prefix}")
                plt.xlabel(var)
                plt.ylabel("Count")
                per_fig_path = os.path.join(fig_dir, f"{var}_hist.png")
                plt.savefig(per_fig_path, bbox_inches="tight")
                plt.close()
                print(f"  saved figure: {os.path.abspath(per_fig_path)}")

                summary_rows.append({
                    "variable": var,
                    "label": label,
                    "type": "numeric",
                    "n_nonnull": int(series_num.shape[0]),
                    "mean": mean_val,
                    "std": std_val,
                    "n_unique_categories": np.nan
                })

    # Global summary
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_csv_path = os.path.join(table_dir, f"{prefix}_summary_overview.csv")
        summary_df.to_csv(summary_csv_path, index=False)
        print(f"[done] Global summary saved: {os.path.abspath(summary_csv_path)}")

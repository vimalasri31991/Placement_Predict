from pathlib import Path
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from .data_loader import load_data

BASE_DIR = Path(__file__).resolve().parent.parent
CHART_DIR = BASE_DIR / "static" / "charts"
CHART_DIR.mkdir(parents=True, exist_ok=True)
CACHE_FILE = BASE_DIR / "static" / "eda_cache.json"

_MEM_CACHE = None


def _set_plot_style():
    sns.set_theme(style="whitegrid", palette="deep")
    plt.rcParams["font.sans-serif"] = ["Segoe UI", "DejaVu Sans", "Arial", "sans-serif"]
    plt.rcParams["axes.edgecolor"] = "#ccd7e6"
    plt.rcParams["axes.linewidth"] = 0.8


def generate_eda_charts(df, force=False):
    """
    Generate all charts for the 15 EDA tasks.
    Saves high quality images in static/charts/.
    """
    _set_plot_style()
    brand_blue = "#1f3a5f"
    brand_light_blue = "#3d6aa3"
    palette_custom = ["#3d6aa3", "#e05638", "#2a9d8f", "#e76f51", "#264653", "#f4a261"]

    # -------------------------------------------------------------
    # Task 3: Missing Values Heatmap
    # -------------------------------------------------------------
    t3_path = CHART_DIR / "task3_missing_heatmap.png"
    if force or not t3_path.exists():
        fig, ax = plt.subplots(figsize=(10, 4.5))
        null_df = df.isnull()
        # Sort columns by missing count for clarity
        cols_sorted = null_df.sum().sort_values(ascending=False).index
        sns.heatmap(null_df[cols_sorted].transpose(), cbar=False, cmap="Blues_r", ax=ax)
        ax.set_title("Task 3: Missing Values Heatmap (Columns Sorted by Null Count)", fontsize=13, fontweight="bold", pad=12, color=brand_blue)
        ax.set_xlabel("Records (0 to 50,000)", fontsize=10)
        ax.set_ylabel("Features", fontsize=10)
        fig.tight_layout()
        fig.savefig(t3_path, dpi=130)
        plt.close(fig)

    # -------------------------------------------------------------
    # Task 5: Target Variable Distribution (PlacementStatus)
    # -------------------------------------------------------------
    t5_path = CHART_DIR / "task5_placement_distribution.png"
    if force or not t5_path.exists():
        fig, ax = plt.subplots(figsize=(7, 4.5))
        counts = df["PlacementStatus"].value_counts().sort_index()
        colors = ["#e05638", "#2a9d8f"]
        bars = ax.bar(["Not Placed (0)", "Placed (1)"], counts.values, color=colors, width=0.45, edgecolor="#1f3a5f", linewidth=1.2)
        for bar in bars:
            h = bar.get_height()
            pct = (h / len(df)) * 100
            ax.text(bar.get_x() + bar.get_width() / 2, h + 800, f"{h:,} ({pct:.1f}%)", ha="center", va="bottom", fontsize=11, fontweight="bold", color="#1f3a5f")
        ax.set_title("Task 5: Target Variable Distribution (PlacementStatus)", fontsize=13, fontweight="bold", pad=12, color=brand_blue)
        ax.set_ylabel("Student Count", fontsize=10)
        ax.set_ylim(0, max(counts.values) * 1.15)
        fig.tight_layout()
        fig.savefig(t5_path, dpi=130)
        plt.close(fig)

    # -------------------------------------------------------------
    # Task 6: Numeric Feature Distributions
    # -------------------------------------------------------------
    t6_path = CHART_DIR / "task6_numeric_distributions.png"
    if force or not t6_path.exists():
        num_features = ["CGPA", "AttendancePercent", "AptitudeTestScore", "CodingTestScore", "SoftSkillsRating", "MockInterviewScore"]
        fig, axes = plt.subplots(2, 3, figsize=(14, 8))
        axes = axes.flatten()
        for idx, col in enumerate(num_features):
            if col in df.columns:
                sns.histplot(df[col].dropna(), kde=True, ax=axes[idx], color="#26466f", bins=25, edgecolor="white")
                mean_val = df[col].mean()
                median_val = df[col].median()
                axes[idx].axvline(mean_val, color="#e05638", linestyle="--", linewidth=1.5, label=f"Mean: {mean_val:.2f}")
                axes[idx].axvline(median_val, color="#2a9d8f", linestyle="-.", linewidth=1.5, label=f"Median: {median_val:.2f}")
                axes[idx].set_title(f"Distribution: {col}", fontsize=11, fontweight="bold", color=brand_blue)
                axes[idx].legend(fontsize=8, loc="upper right")
        fig.suptitle("Task 6: Numeric Feature Distributions (Histograms with KDE)", fontsize=14, fontweight="bold", y=1.02, color=brand_blue)
        fig.tight_layout()
        fig.savefig(t6_path, dpi=130)
        plt.close(fig)

    # -------------------------------------------------------------
    # Task 7: Outlier Detection (Boxplots)
    # -------------------------------------------------------------
    t7_path = CHART_DIR / "task7_outlier_boxplots.png"
    if force or not t7_path.exists():
        box_cols = ["CGPA", "AptitudeTestScore", "CodingTestScore", "SoftSkillsRating", "MockInterviewScore", "Salary Package"]
        fig, axes = plt.subplots(2, 3, figsize=(14, 8))
        axes = axes.flatten()
        for idx, col in enumerate(box_cols):
            if col in df.columns:
                sns.boxplot(y=df[col].dropna(), ax=axes[idx], color="#90b4ce", flierprops=dict(marker="o", markersize=3, markerfacecolor="#e05638", alpha=0.5))
                axes[idx].set_title(f"Boxplot: {col}", fontsize=11, fontweight="bold", color=brand_blue)
                axes[idx].set_ylabel(col, fontsize=10)
        fig.suptitle("Task 7: Outlier Detection via Boxplots for Key Numeric Features", fontsize=14, fontweight="bold", y=1.02, color=brand_blue)
        fig.tight_layout()
        fig.savefig(t7_path, dpi=130)
        plt.close(fig)

    # -------------------------------------------------------------
    # Task 8: Correlation Analysis (Heatmap + Target Correlation)
    # -------------------------------------------------------------
    t8_path = CHART_DIR / "task8_correlation_heatmap.png"
    if force or not t8_path.exists():
        numeric_df = df.select_dtypes(include="number").drop(columns=["StudentID"], errors="ignore")
        fig = plt.figure(figsize=(16, 7))
        gs = fig.add_gridspec(1, 2, width_ratios=[1.3, 1])

        # Subplot 1: Correlation with PlacementStatus
        ax1 = fig.add_subplot(gs[0])
        corr_series = numeric_df.corr()["PlacementStatus"].drop("PlacementStatus", errors="ignore").sort_values()
        bar_colors = ["#2a9d8f" if v >= 0 else "#e05638" for v in corr_series.values]
        ax1.barh(corr_series.index, corr_series.values, color=bar_colors, edgecolor="#1f3a5f", height=0.65)
        ax1.set_title("Correlation of Features with PlacementStatus", fontsize=12, fontweight="bold", color=brand_blue)
        ax1.set_xlabel("Pearson Correlation Coefficient")
        ax1.grid(axis="x", linestyle="--", alpha=0.7)

        # Subplot 2: Correlation Heatmap for Core Features
        ax2 = fig.add_subplot(gs[1])
        core_cols = ["CGPA", "AttendancePercent", "AptitudeTestScore", "CodingTestScore", "SoftSkillsRating", "MockInterviewScore", "SGPA_Sem8", "Salary Package", "PlacementStatus"]
        corr_matrix = df[core_cols].corr()
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax2, cbar_kws={"shrink": 0.8}, annot_kws={"size": 8})
        ax2.set_title("Core Features Correlation Heatmap", fontsize=12, fontweight="bold", color=brand_blue)

        fig.suptitle("Task 8: Correlation Analysis (Target Association & Feature Intercorrelations)", fontsize=14, fontweight="bold", y=1.02, color=brand_blue)
        fig.tight_layout()
        fig.savefig(t8_path, dpi=130)
        plt.close(fig)

    # -------------------------------------------------------------
    # Task 9: Relationship Plots (Regression Scatter Plots)
    # -------------------------------------------------------------
    t9_path = CHART_DIR / "task9_relationship_plots.png"
    if force or not t9_path.exists():
        sample_df = df.sample(n=min(3000, len(df)), random_state=42)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

        # Panel 1: CGPA vs Salary Package (Placed students)
        placed_sample = sample_df[sample_df["PlacementStatus"] == 1]
        sns.regplot(data=placed_sample, x="CGPA", y="Salary Package", ax=ax1,
                    scatter_kws={"alpha": 0.4, "color": "#3d6aa3", "s": 18},
                    line_kws={"color": "#e05638", "linewidth": 2.2})
        ax1.set_title("CGPA vs Salary Package (Placed Students)", fontsize=12, fontweight="bold", color=brand_blue)
        ax1.set_xlabel("Cumulative GPA (CGPA)")
        ax1.set_ylabel("Salary Package (LPA)")

        # Panel 2: Aptitude vs Coding Test Score
        sns.regplot(data=sample_df, x="AptitudeTestScore", y="CodingTestScore", ax=ax2,
                    scatter_kws={"alpha": 0.4, "color": "#2a9d8f", "s": 18},
                    line_kws={"color": "#e76f51", "linewidth": 2.2})
        ax2.set_title("Aptitude Test Score vs Coding Test Score", fontsize=12, fontweight="bold", color=brand_blue)
        ax2.set_xlabel("Aptitude Test Score")
        ax2.set_ylabel("Coding Test Score")

        fig.suptitle("Task 9: Relationship Plots (Bivariate Regression Scatter Plots)", fontsize=14, fontweight="bold", y=1.02, color=brand_blue)
        fig.tight_layout()
        fig.savefig(t9_path, dpi=130)
        plt.close(fig)

    # -------------------------------------------------------------
    # Task 10: Categorical Feature Counts
    # -------------------------------------------------------------
    t10_path = CHART_DIR / "task10_categorical_counts.png"
    if force or not t10_path.exists():
        cat_features = ["Gender", "City", "CollegeTier", "Stream", "Specialisation", "Hostel", "HistoryOfBacklogs", "CGPA_Tier"]
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()
        for idx, col in enumerate(cat_features):
            if col in df.columns:
                val_counts = df[col].value_counts()
                sns.barplot(x=val_counts.index, y=val_counts.values, ax=axes[idx], color="#3d6aa3")
                axes[idx].set_title(f"Counts: {col}", fontsize=11, fontweight="bold", color=brand_blue)
                axes[idx].set_ylabel("Students")
                axes[idx].tick_params(axis="x", rotation=30)
        fig.suptitle("Task 10: Categorical Feature Counts (Univariate Analysis)", fontsize=14, fontweight="bold", y=1.02, color=brand_blue)
        fig.tight_layout()
        fig.savefig(t10_path, dpi=130)
        plt.close(fig)

    # -------------------------------------------------------------
    # Task 11: Gender vs Placement Status
    # -------------------------------------------------------------
    t11_path = CHART_DIR / "task11_gender_vs_placement.png"
    if force or not t11_path.exists():
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
        # Grouped count
        gender_placed = pd.crosstab(df["Gender"], df["PlacementStatus"])
        gender_placed.columns = ["Not Placed", "Placed"]
        gender_placed.plot(kind="bar", stacked=False, color=["#e05638", "#2a9d8f"], ax=ax1, width=0.5, edgecolor="#1f3a5f")
        ax1.set_title("Student Counts by Gender & Placement Status", fontsize=11, fontweight="bold", color=brand_blue)
        ax1.set_ylabel("Count")
        ax1.tick_params(axis="x", rotation=0)

        # Placement rate %
        placement_rate_gender = df.groupby("Gender")["PlacementStatus"].mean() * 100
        bars = ax2.bar(placement_rate_gender.index, placement_rate_gender.values, color=["#3d6aa3", "#e76f51"], width=0.4, edgecolor="#1f3a5f")
        for bar in bars:
            h = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2, h + 1.5, f"{h:.1f}%", ha="center", va="bottom", fontweight="bold", color=brand_blue)
        ax2.set_title("Placement Rate (%) by Gender", fontsize=11, fontweight="bold", color=brand_blue)
        ax2.set_ylabel("Placement Rate %")
        ax2.set_ylim(0, 100)

        fig.suptitle("Task 11: Gender vs Placement Status Breakdown (Bivariate Analysis)", fontsize=14, fontweight="bold", y=1.02, color=brand_blue)
        fig.tight_layout()
        fig.savefig(t11_path, dpi=130)
        plt.close(fig)

    # -------------------------------------------------------------
    # Task 12: College Tier / Stream vs Placement Status
    # -------------------------------------------------------------
    t12_path = CHART_DIR / "task12_tier_stream_vs_placement.png"
    if force or not t12_path.exists():
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # College Tier placement rate
        tier_rate = df.groupby("CollegeTier")["PlacementStatus"].mean().sort_values(ascending=False) * 100
        bars1 = ax1.bar(tier_rate.index, tier_rate.values, color=["#1f3a5f", "#3d6aa3", "#55769d"], width=0.5, edgecolor="#1f3a5f")
        for bar in bars1:
            h = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2, h + 1.5, f"{h:.1f}%", ha="center", va="bottom", fontweight="bold", color=brand_blue)
        ax1.set_title("Placement Rate by College Tier", fontsize=12, fontweight="bold", color=brand_blue)
        ax1.set_ylabel("Placement Rate %")
        ax1.set_ylim(0, 100)

        # Stream placement rate
        stream_rate = df.groupby("Stream")["PlacementStatus"].mean().sort_values(ascending=False) * 100
        bars2 = ax2.bar(stream_rate.index, stream_rate.values, color=["#2a9d8f", "#264653", "#e76f51", "#f4a261", "#e05638"], width=0.55, edgecolor="#1f3a5f")
        for bar in bars2:
            h = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2, h + 1.5, f"{h:.1f}%", ha="center", va="bottom", fontweight="bold", color=brand_blue)
        ax2.set_title("Placement Rate by Academic Stream", fontsize=12, fontweight="bold", color=brand_blue)
        ax2.set_ylabel("Placement Rate %")
        ax2.set_ylim(0, 100)

        fig.suptitle("Task 12: College Tier and Academic Stream vs Placement Status", fontsize=14, fontweight="bold", y=1.02, color=brand_blue)
        fig.tight_layout()
        fig.savefig(t12_path, dpi=130)
        plt.close(fig)

    # -------------------------------------------------------------
    # Task 13: SGPA Trend Across Semesters
    # -------------------------------------------------------------
    t13_path = CHART_DIR / "task13_sgpa_trend.png"
    if force or not t13_path.exists():
        sem_cols = [f"SGPA_Sem{i}" for i in range(1, 9)]
        sem_labels = [f"Sem {i}" for i in range(1, 9)]
        fig, ax = plt.subplots(figsize=(10, 5))

        mean_overall = df[sem_cols].mean().values
        mean_placed = df[df["PlacementStatus"] == 1][sem_cols].mean().values
        mean_unplaced = df[df["PlacementStatus"] == 0][sem_cols].mean().values

        ax.plot(sem_labels, mean_placed, marker="o", linewidth=2.5, color="#2a9d8f", label="Placed Students (1)")
        ax.plot(sem_labels, mean_overall, marker="s", linewidth=2, linestyle="--", color="#1f3a5f", label="Overall Average")
        ax.plot(sem_labels, mean_unplaced, marker="^", linewidth=2.5, color="#e05638", label="Not Placed (0)")

        for x, y in zip(sem_labels, mean_placed):
            ax.text(x, y + 0.12, f"{y:.2f}", ha="center", fontsize=9, fontweight="bold", color="#2a9d8f")
        for x, y in zip(sem_labels, mean_unplaced):
            ax.text(x, y - 0.22, f"{y:.2f}", ha="center", fontsize=9, fontweight="bold", color="#e05638")

        ax.set_title("Task 13: Average SGPA Progression Across Semesters (Sem1 – Sem8)", fontsize=13, fontweight="bold", pad=12, color=brand_blue)
        ax.set_ylabel("Average SGPA", fontsize=10)
        ax.set_ylim(min(mean_unplaced) - 0.5, max(mean_placed) + 0.5)
        ax.legend(loc="lower right", fontsize=10)
        fig.tight_layout()
        fig.savefig(t13_path, dpi=130)
        plt.close(fig)

    # -------------------------------------------------------------
    # Task 14: Salary Package Analysis
    # -------------------------------------------------------------
    t14_path = CHART_DIR / "task14_salary_analysis.png"
    if force or not t14_path.exists():
        placed_df = df[df["PlacementStatus"] == 1]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Panel 1: Salary Distribution for Placed Students
        sns.histplot(placed_df["Salary Package"], kde=True, ax=ax1, color="#3d6aa3", bins=30, edgecolor="white")
        med_sal = placed_df["Salary Package"].median()
        mean_sal = placed_df["Salary Package"].mean()
        ax1.axvline(med_sal, color="#e05638", linestyle="--", linewidth=2, label=f"Median: {med_sal:.2f} LPA")
        ax1.axvline(mean_sal, color="#2a9d8f", linestyle="-.", linewidth=2, label=f"Mean: {mean_sal:.2f} LPA")
        ax1.set_title("Salary Package Distribution (Placed Students)", fontsize=11, fontweight="bold", color=brand_blue)
        ax1.set_xlabel("Salary Package (LPA)")
        ax1.legend()

        # Panel 2: Salary by College Tier
        sns.boxplot(data=placed_df, x="CollegeTier", y="Salary Package", ax=ax2, color="#90b4ce")
        ax2.set_title("Salary Package by College Tier", fontsize=11, fontweight="bold", color=brand_blue)
        ax2.set_ylabel("Salary Package (LPA)")

        fig.suptitle("Task 14: Salary Package Analysis (Univariate & Bivariate Distribution)", fontsize=14, fontweight="bold", y=1.02, color=brand_blue)
        fig.tight_layout()
        fig.savefig(t14_path, dpi=130)
        plt.close(fig)

    # -------------------------------------------------------------
    # Task 15: Pairplot
    # -------------------------------------------------------------
    t15_path = CHART_DIR / "task15_pairplot.png"
    if force or not t15_path.exists():
        sample_placed = df[df["PlacementStatus"] == 1].sample(n=min(1000, (df["PlacementStatus"] == 1).sum()), random_state=42)
        sample_unplaced = df[df["PlacementStatus"] == 0].sample(n=min(1000, (df["PlacementStatus"] == 0).sum()), random_state=42)
        sample_pair = pd.concat([sample_placed, sample_unplaced])

        pair_cols = ["CGPA", "AptitudeTestScore", "CodingTestScore", "MockInterviewScore", "PlacementStatus"]
        pair_df = sample_pair[pair_cols].dropna().copy()
        pair_df["PlacementStatus"] = pair_df["PlacementStatus"].map({1: "Placed", 0: "Not Placed"})

        g = sns.pairplot(pair_df, hue="PlacementStatus", palette={"Placed": "#2a9d8f", "Not Placed": "#e05638"},
                         plot_kws={"alpha": 0.45, "s": 15}, diag_kind="kde", corner=False)
        g.fig.suptitle("Task 15: Pairplot of Core Evaluation Scores by Placement Status", y=1.02, fontsize=14, fontweight="bold", color=brand_blue)
        g.savefig(t15_path, dpi=110, bbox_inches="tight")
        plt.close(g.fig)


def run_eda(force_refresh=False):
    """
    Execute all 15 tasks on placement_eda and return structured summary and charts.
    Cached in-memory and on disk for sub-millisecond response.
    """
    global _MEM_CACHE
    if not force_refresh and _MEM_CACHE is not None:
        return _MEM_CACHE

    df = load_data().copy()

    # Pre-generate all charts if missing
    generate_eda_charts(df, force=force_refresh)

    # -------------------------------------------------------------
    # Task 1: Load Data
    # -------------------------------------------------------------
    task1 = {
        "shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
        "first_5_rows": df.head(5).to_dict(orient="records"),
        "columns": list(df.columns)
    }

    # -------------------------------------------------------------
    # Task 2: Basic Info / Structure (.info(), dtypes, .describe())
    # -------------------------------------------------------------
    column_info = []
    missing_series = df.isnull().sum()
    for col in df.columns:
        column_info.append({
            "column": col,
            "dtype": str(df[col].dtype),
            "non_null_count": int(df[col].notnull().sum()),
            "null_count": int(missing_series[col]),
            "unique_count": int(df[col].nunique(dropna=True))
        })

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    numeric_describe = (
        df[numeric_cols]
        .describe()
        .transpose()
        .round(3)
        .reset_index()
        .rename(columns={"index": "feature"})
        .to_dict(orient="records")
    )

    categorical_describe = (
        df[categorical_cols]
        .describe()
        .transpose()
        .reset_index()
        .rename(columns={"index": "feature"})
        .to_dict(orient="records")
    )

    task2 = {
        "column_info": column_info,
        "numeric_describe": numeric_describe,
        "categorical_describe": categorical_describe,
        "num_numeric": len(numeric_cols),
        "num_categorical": len(categorical_cols)
    }

    # -------------------------------------------------------------
    # Task 3: Missing Values
    # -------------------------------------------------------------
    missing_table = []
    for col in df.columns:
        m_count = int(missing_series[col])
        if m_count > 0:
            m_pct = round((m_count / len(df)) * 100, 2)
            med_fill = round(float(df[col].median()), 2) if np.issubdtype(df[col].dtype, np.number) else "Mode"
            missing_table.append({
                "column": col,
                "missing_count": m_count,
                "missing_percent": m_pct,
                "median_fill": med_fill
            })

    task3 = {
        "missing_table": missing_table,
        "total_missing": int(missing_series.sum()),
        "chart": "task3_missing_heatmap.png"
    }

    # -------------------------------------------------------------
    # Task 4: Duplicate Rows
    # -------------------------------------------------------------
    dup_count = int(df.duplicated().sum())
    task4 = {
        "duplicate_count": dup_count,
        "duplicate_percent": round((dup_count / len(df)) * 100, 2),
        "status": "Clean (No duplicate records found)" if dup_count == 0 else f"{dup_count} duplicates detected"
    }

    # -------------------------------------------------------------
    # Task 5: Target Variable Distribution
    # -------------------------------------------------------------
    t_counts = df["PlacementStatus"].value_counts().to_dict()
    task5 = {
        "placed_count": int(t_counts.get(1, 0)),
        "not_placed_count": int(t_counts.get(0, 0)),
        "placed_percent": round(float(t_counts.get(1, 0) / len(df)) * 100, 2),
        "not_placed_percent": round(float(t_counts.get(0, 0) / len(df)) * 100, 2),
        "chart": "task5_placement_distribution.png"
    }

    # -------------------------------------------------------------
    # Task 6: Numeric Feature Distributions
    # -------------------------------------------------------------
    key_numeric = ["CGPA", "AttendancePercent", "AptitudeTestScore", "CodingTestScore", "SoftSkillsRating", "MockInterviewScore"]
    num_stats = []
    for col in key_numeric:
        if col in df.columns:
            s = df[col].dropna()
            num_stats.append({
                "feature": col,
                "mean": round(float(s.mean()), 2),
                "std": round(float(s.std()), 2),
                "min": round(float(s.min()), 2),
                "median": round(float(s.median()), 2),
                "max": round(float(s.max()), 2),
                "skew": round(float(s.skew()), 2)
            })
    task6 = {
        "stats": num_stats,
        "chart": "task6_numeric_distributions.png"
    }

    # -------------------------------------------------------------
    # Task 7: Outlier Detection (Boxplots & IQR)
    # -------------------------------------------------------------
    outlier_table = []
    for col in ["CGPA", "AptitudeTestScore", "CodingTestScore", "SoftSkillsRating", "MockInterviewScore", "Salary Package"]:
        if col in df.columns:
            s = df[col].dropna()
            q1 = float(s.quantile(0.25))
            q3 = float(s.quantile(0.75))
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            outliers = int(((s < lower) | (s > upper)).sum())
            outlier_table.append({
                "feature": col,
                "q1": round(q1, 2),
                "q3": round(q3, 2),
                "iqr": round(iqr, 2),
                "lower_bound": round(lower, 2),
                "upper_bound": round(upper, 2),
                "outlier_count": outliers,
                "outlier_percent": round((outliers / len(s)) * 100, 2)
            })
    task7 = {
        "table": outlier_table,
        "chart": "task7_outlier_boxplots.png"
    }

    # -------------------------------------------------------------
    # Task 8: Correlation Analysis
    # -------------------------------------------------------------
    corr_series = df[numeric_cols].drop(columns=["StudentID"], errors="ignore").corr()["PlacementStatus"].sort_values(ascending=False)
    corr_list = []
    for feat, val in corr_series.items():
        if feat != "PlacementStatus":
            corr_list.append({
                "feature": feat,
                "correlation": round(float(val), 4),
                "strength": "Strong Positive" if val >= 0.6 else "Moderate Positive" if val >= 0.3 else "Weak Positive" if val > 0 else "Negative"
            })
    task8 = {
        "correlations": corr_list[:10],
        "chart": "task8_correlation_heatmap.png"
    }

    # -------------------------------------------------------------
    # Task 9: Relationship Plots
    # -------------------------------------------------------------
    placed_df = df[df["PlacementStatus"] == 1]
    cgpa_salary_corr = round(float(placed_df[["CGPA", "Salary Package"]].dropna().corr().iloc[0, 1]), 4)
    apt_code_corr = round(float(df[["AptitudeTestScore", "CodingTestScore"]].dropna().corr().iloc[0, 1]), 4)
    task9 = {
        "cgpa_salary_corr": cgpa_salary_corr,
        "apt_code_corr": apt_code_corr,
        "chart": "task9_relationship_plots.png"
    }

    # -------------------------------------------------------------
    # Task 10: Categorical Feature Counts
    # -------------------------------------------------------------
    cat_counts = {}
    for col in categorical_cols:
        cat_counts[col] = df[col].value_counts().to_dict()
    task10 = {
        "counts": cat_counts,
        "chart": "task10_categorical_counts.png"
    }

    # -------------------------------------------------------------
    # Task 11: Gender vs Placement Status
    # -------------------------------------------------------------
    g_tab = pd.crosstab(df["Gender"], df["PlacementStatus"], margins=True)
    gender_table = []
    for g in ["Male", "Female"]:
        if g in g_tab.index:
            not_p = int(g_tab.loc[g, 0])
            p = int(g_tab.loc[g, 1])
            tot = not_p + p
            rate = round((p / tot) * 100, 2)
            gender_table.append({
                "gender": g,
                "placed": p,
                "not_placed": not_p,
                "total": tot,
                "placement_rate": rate
            })
    task11 = {
        "table": gender_table,
        "chart": "task11_gender_vs_placement.png"
    }

    # -------------------------------------------------------------
    # Task 12: College Tier / Stream vs Placement Status
    # -------------------------------------------------------------
    tier_table = []
    for tier in sorted(df["CollegeTier"].dropna().unique()):
        sub = df[df["CollegeTier"] == tier]
        rate = round(float(sub["PlacementStatus"].mean() * 100), 2)
        tier_table.append({
            "tier": tier,
            "total": int(len(sub)),
            "placed": int(sub["PlacementStatus"].sum()),
            "placement_rate": rate
        })

    stream_table = []
    for stream in sorted(df["Stream"].dropna().unique()):
        sub = df[df["Stream"] == stream]
        rate = round(float(sub["PlacementStatus"].mean() * 100), 2)
        stream_table.append({
            "stream": stream,
            "total": int(len(sub)),
            "placed": int(sub["PlacementStatus"].sum()),
            "placement_rate": rate
        })

    task12 = {
        "tier_table": tier_table,
        "stream_table": stream_table,
        "chart": "task12_tier_stream_vs_placement.png"
    }

    # -------------------------------------------------------------
    # Task 13: SGPA Trend Across Semesters
    # -------------------------------------------------------------
    sem_cols = [f"SGPA_Sem{i}" for i in range(1, 9)]
    sgpa_trend_table = []
    for idx, sc in enumerate(sem_cols, start=1):
        sgpa_trend_table.append({
            "semester": f"Semester {idx}",
            "overall_avg": round(float(df[sc].mean()), 2),
            "placed_avg": round(float(df[df["PlacementStatus"] == 1][sc].mean()), 2),
            "not_placed_avg": round(float(df[df["PlacementStatus"] == 0][sc].mean()), 2)
        })
    task13 = {
        "trend_table": sgpa_trend_table,
        "chart": "task13_sgpa_trend.png"
    }

    # -------------------------------------------------------------
    # Task 14: Salary Package Analysis
    # -------------------------------------------------------------
    sal_placed = placed_df["Salary Package"].dropna()
    tier_salaries = []
    for tier in sorted(df["CollegeTier"].dropna().unique()):
        t_sal = placed_df[placed_df["CollegeTier"] == tier]["Salary Package"].dropna()
        tier_salaries.append({
            "tier": tier,
            "median": round(float(t_sal.median()), 2) if not t_sal.empty else 0,
            "mean": round(float(t_sal.mean()), 2) if not t_sal.empty else 0,
            "min": round(float(t_sal.min()), 2) if not t_sal.empty else 0,
            "max": round(float(t_sal.max()), 2) if not t_sal.empty else 0
        })
    task14 = {
        "placed_mean": round(float(sal_placed.mean()), 2),
        "placed_median": round(float(sal_placed.median()), 2),
        "placed_min": round(float(sal_placed.min()), 2),
        "placed_max": round(float(sal_placed.max()), 2),
        "tier_salaries": tier_salaries,
        "chart": "task14_salary_analysis.png"
    }

    # -------------------------------------------------------------
    # Task 15: Pairplot
    # -------------------------------------------------------------
    task15 = {
        "features": ["CGPA", "AptitudeTestScore", "CodingTestScore", "MockInterviewScore"],
        "target": "PlacementStatus",
        "chart": "task15_pairplot.png"
    }

    result = {
        "task1": task1,
        "task2": task2,
        "task3": task3,
        "task4": task4,
        "task5": task5,
        "task6": task6,
        "task7": task7,
        "task8": task8,
        "task9": task9,
        "task10": task10,
        "task11": task11,
        "task12": task12,
        "task13": task13,
        "task14": task14,
        "task15": task15,
        # Backward compatibility fields
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "placement_counts": t_counts,
        "charts": [
            "task3_missing_heatmap.png",
            "task5_placement_distribution.png",
            "task6_numeric_distributions.png",
            "task7_outlier_boxplots.png",
            "task8_correlation_heatmap.png",
            "task9_relationship_plots.png",
            "task10_categorical_counts.png",
            "task11_gender_vs_placement.png",
            "task12_tier_stream_vs_placement.png",
            "task13_sgpa_trend.png",
            "task14_salary_analysis.png",
            "task15_pairplot.png"
        ]
    }

    _MEM_CACHE = result
    return result
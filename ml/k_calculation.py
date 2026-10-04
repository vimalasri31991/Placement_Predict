from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from .model_data import get_model_data

BASE_DIR = Path(__file__).resolve().parent.parent
CHART_DIR = BASE_DIR / "static" / "charts"
CHART_DIR.mkdir(parents=True, exist_ok=True)

_CACHED_K_BASE = None


def get_base_k_data(max_k=10):
    global _CACHED_K_BASE
    if _CACHED_K_BASE is not None:
        return _CACHED_K_BASE

    (
        X_train,
        X_test,
        y_train,
        y_test,
        _
    ) = get_model_data(
        scale_numeric=True
    )

    np.random.seed(42)
    sample_size = min(3000, len(X_train))
    sample_indices = np.random.choice(len(X_train), size=sample_size, replace=False)
    X_sample = X_train[sample_indices]

    rows = []
    k_range = list(range(2, max_k + 1))
    inertias = []
    silhouettes = []

    for k in k_range:
        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )
        labels = model.fit_predict(X_sample)
        sil = float(silhouette_score(X_sample, labels))
        inert = float(model.inertia_)

        inertias.append(inert)
        silhouettes.append(sil)

        reduction = 0.0
        if len(inertias) > 1:
            reduction = round(((inertias[-2] - inert) / inertias[-2]) * 100, 2)

        rows.append({
            "k": k,
            "inertia": inert,
            "silhouette": sil,
            "reduction_pct": reduction
        })

    best_sil_idx = int(np.argmax(silhouettes))
    best_k_sil = k_range[best_sil_idx]

    diffs = np.diff(inertias)
    diffs2 = np.diff(diffs)
    elbow_idx = int(np.argmax(diffs2)) if len(diffs2) > 0 else 1
    best_k_elbow = k_range[elbow_idx + 1]

    _CACHED_K_BASE = {
        "rows": rows,
        "k_range": k_range,
        "inertias": inertias,
        "silhouettes": silhouettes,
        "optimal_k_elbow": best_k_elbow,
        "optimal_k_silhouette": best_k_sil
    }
    return _CACHED_K_BASE


def calculate_k_values(max_k=10, selected_k=None, method_type="elbow"):
    """
    Calculate or retrieve K values analysis with manual K selection supported across all methods.
    """
    base = get_base_k_data(max_k)
    k_range = base["k_range"]
    inertias = base["inertias"]
    silhouettes = base["silhouettes"]

    if selected_k is None:
        selected_k = base["optimal_k_elbow"] if method_type == "elbow" else base["optimal_k_silhouette"]
    else:
        try:
            selected_k = int(selected_k)
            if selected_k not in k_range:
                selected_k = k_range[0]
        except (ValueError, TypeError):
            selected_k = 3

    # Generate or retrieve specific chart with selected K highlighted
    elbow_chart = f"kmeans_elbow_k{selected_k}.png"
    sil_chart = f"kmeans_silhouette_k{selected_k}.png"

    # Elbow chart
    if not (CHART_DIR / elbow_chart).exists():
        fig, ax = plt.subplots(figsize=(8.5, 4.8))
        ax.plot(k_range, inertias, marker="o", color="#1f3a5f", linewidth=2.5, markersize=7, label="Inertia (WCSS)")
        sel_idx = k_range.index(selected_k)
        ax.scatter([selected_k], [inertias[sel_idx]], color="#e05638", s=150, zorder=5, label=f"Selected K = {selected_k}")
        ax.axvline(selected_k, color="#e05638", linestyle="--", alpha=0.7)

        for k_val, in_val in zip(k_range, inertias):
            ax.annotate(f"{in_val:,.0f}", (k_val, in_val), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=8.5, fontweight="bold", color="#1f3a5f")

        ax.set_title(f"K-Means Elbow Method: WCSS vs K (Selected K = {selected_k})", fontsize=12, fontweight="bold", pad=12, color="#1f3a5f")
        ax.set_xlabel("Number of Clusters (K)", fontsize=10)
        ax.set_ylabel("Inertia (Within-Cluster Sum of Squares)", fontsize=10)
        ax.set_xticks(k_range)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(loc="upper right", fontsize=9.5)
        fig.tight_layout()
        fig.savefig(CHART_DIR / elbow_chart, dpi=130)
        plt.close(fig)

    # Silhouette chart
    if not (CHART_DIR / sil_chart).exists():
        fig, ax = plt.subplots(figsize=(8.5, 4.8))
        ax.plot(k_range, silhouettes, marker="s", color="#2a9d8f", linewidth=2.5, markersize=7, label="Silhouette Score")
        sel_idx = k_range.index(selected_k)
        ax.scatter([selected_k], [silhouettes[sel_idx]], color="#e76f51", s=150, zorder=5, label=f"Selected K = {selected_k}")
        ax.axvline(selected_k, color="#e76f51", linestyle="--", alpha=0.7)

        for k_val, sil_val in zip(k_range, silhouettes):
            ax.annotate(f"{sil_val:.4f}", (k_val, sil_val), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=8.5, fontweight="bold", color="#264653")

        ax.set_title(f"K-Means Silhouette Analysis vs K (Selected K = {selected_k})", fontsize=12, fontweight="bold", pad=12, color="#1f3a5f")
        ax.set_xlabel("Number of Clusters (K)", fontsize=10)
        ax.set_ylabel("Average Silhouette Coefficient", fontsize=10)
        ax.set_xticks(k_range)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(loc="upper right", fontsize=9.5)
        fig.tight_layout()
        fig.savefig(CHART_DIR / sil_chart, dpi=130)
        plt.close(fig)

    # Retrieve selected row metrics
    selected_row = next((r for r in base["rows"] if r["k"] == selected_k), base["rows"][0])

    return {
        "rows": base["rows"],
        "k_options": k_range,
        "selected_k": selected_k,
        "selected_inertia": selected_row["inertia"],
        "selected_silhouette": selected_row["silhouette"],
        "selected_reduction": selected_row["reduction_pct"],
        "optimal_k_elbow": base["optimal_k_elbow"],
        "optimal_k_silhouette": base["optimal_k_silhouette"],
        "elbow_chart": elbow_chart,
        "silhouette_chart": sil_chart
    }
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from .model_data import get_model_data

BASE_DIR = Path(__file__).resolve().parent.parent
CHART_DIR = BASE_DIR / "static" / "charts"
CHART_DIR.mkdir(parents=True, exist_ok=True)

_KMEANS_CACHE = {}


def run_kmeans(n_clusters=3, force=False):
    """
    Run K-Means clustering for a user-specified K (Manual K Selection).
    Computes cluster sizes, inertia, silhouette score, feature averages,
    and generates a 2D PCA cluster projection chart.
    """
    try:
        n_clusters = int(n_clusters)
    except (ValueError, TypeError):
        n_clusters = 3

    if not force and n_clusters in _KMEANS_CACHE:
        return _KMEANS_CACHE[n_clusters]

    (
        X_train,
        X_test,
        y_train,
        y_test,
        feature_names
    ) = get_model_data(
        scale_numeric=True
    )

    # Stratified sample for fast fitting and plotting
    np.random.seed(42)
    sample_size = min(3000, len(X_train))
    sample_idx = np.random.choice(len(X_train), size=sample_size, replace=False)
    X_sample = X_train[sample_idx]
    y_sample = y_train.iloc[sample_idx] if hasattr(y_train, "iloc") else y_train[sample_idx]

    model = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(X_sample)
    sil = float(silhouette_score(X_sample, labels))
    inertia = float(model.inertia_)

    # Cluster counts and placement rates
    counts = {}
    cluster_profiles = []
    colors = ["#1f3a5f", "#2a9d8f", "#e05638", "#f4a261", "#e76f51", "#7209b7", "#4361ee", "#4cc9f0"]

    for k in range(n_clusters):
        mask = (labels == k)
        c_count = int(mask.sum())
        c_pct = round((c_count / sample_size) * 100, 2)
        counts[k] = c_count

        # Placement rate within cluster
        c_y = y_sample[mask] if isinstance(y_sample, np.ndarray) else y_sample.iloc[mask]
        placement_rate = round(float(c_y.mean()) * 100, 2) if len(c_y) > 0 else 0.0

        cluster_profiles.append({
            "cluster": k,
            "count": c_count,
            "percent": c_pct,
            "placement_rate": placement_rate,
            "color": colors[k % len(colors)]
        })

    # 2D PCA for cluster scatter visualization
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_sample)

    chart_name = f"kmeans_clusters_k{n_clusters}.png"
    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    for k in range(n_clusters):
        mask = (labels == k)
        ax.scatter(
            X_pca[mask, 0],
            X_pca[mask, 1],
            c=colors[k % len(colors)],
            label=f"Cluster {k} ({counts[k]} students)",
            alpha=0.6,
            s=22,
            edgecolor="none"
        )

    # Plot centroids in PCA space
    pca_centroids = pca.transform(model.cluster_centers_)
    ax.scatter(
        pca_centroids[:, 0],
        pca_centroids[:, 1],
        marker="X",
        s=160,
        c="yellow",
        edgecolor="black",
        linewidth=1.5,
        label="Centroids"
    )

    ax.set_title(f"K-Means Clustering (K = {n_clusters}): 2D PCA Projection", fontsize=12, fontweight="bold", pad=12, color="#1f3a5f")
    ax.set_xlabel(f"PCA Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)", fontsize=10)
    ax.set_ylabel(f"PCA Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)", fontsize=10)
    ax.legend(loc="upper right", fontsize=8.5, framealpha=0.9)
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(CHART_DIR / chart_name, dpi=130)
    plt.close(fig)

    result = {
        "clusters": n_clusters,
        "inertia": inertia,
        "silhouette": sil,
        "cluster_counts": counts,
        "cluster_profiles": cluster_profiles,
        "chart": chart_name,
        "k_options": [2, 3, 4, 5, 6, 7, 8]
    }

    _KMEANS_CACHE[n_clusters] = result
    return result
from flask import Flask, render_template, request, redirect, url_for
from functools import lru_cache

# ============================================================
# ML IMPORTS
# ============================================================

from ml.data_loader import get_data_summary
from ml.eda import run_eda
from ml.preprocessing import run_preprocessing

from ml.linear_regression import run_linear_regression
from ml.logistic_regression import run_logistic_regression
from ml.regularisation import run_regularisation
from ml.decision_tree import run_decision_tree
from ml.cost_complexity import run_cost_complexity
from ml.random_forest import run_random_forest
from ml.ada_boost import run_adaboost
from ml.gradient_boost import run_gradient_boosting
from ml.xg_boost import run_xgboost

from ml.k_calculation import calculate_k_values
from ml.kmeanfinal import run_kmeans
from ml.hierarchical import run_hierarchical
from ml.dbscan import run_dbscan
from ml.anomaly import run_anomaly_detection

app = Flask(__name__)


# ============================================================
# CACHE FUNCTIONS
# ============================================================

@lru_cache(maxsize=1)
def cached_data_summary():
    return get_data_summary()


@lru_cache(maxsize=1)
def cached_eda():
    return run_eda()


@lru_cache(maxsize=1)
def cached_preprocessing():
    return run_preprocessing()


@lru_cache(maxsize=1)
def cached_linear_regression():
    return run_linear_regression()


@lru_cache(maxsize=1)
def cached_logistic_regression():
    return run_logistic_regression()


@lru_cache(maxsize=1)
def cached_regularisation():
    return run_regularisation()


@lru_cache(maxsize=1)
def cached_decision_tree():
    return run_decision_tree()


@lru_cache(maxsize=1)
def cached_cost_complexity():
    return run_cost_complexity()


@lru_cache(maxsize=1)
def cached_random_forest():
    return run_random_forest()


@lru_cache(maxsize=1)
def cached_adaboost():
    return run_adaboost()


@lru_cache(maxsize=1)
def cached_gradient_boosting():
    return run_gradient_boosting()


@lru_cache(maxsize=1)
def cached_xgboost():
    return run_xgboost()


@lru_cache(maxsize=1)
def cached_k_calculation():
    return calculate_k_values()


@lru_cache(maxsize=16)
def cached_kmeans(n_clusters=3):
    return run_kmeans(n_clusters=n_clusters)


@lru_cache(maxsize=16)
def cached_hierarchical(linkage_method="ward", n_clusters=3):
    return run_hierarchical(linkage_method=linkage_method, n_clusters=n_clusters)


@lru_cache(maxsize=4)
def cached_dbscan(preset="medium"):
    return run_dbscan(preset=preset)


@lru_cache(maxsize=4)
def cached_anomaly(method="isolation-forest"):
    return run_anomaly_detection(method=method)


# ============================================================
# DASHBOARD
# ============================================================

@app.route("/")
def index():
    return render_template("index.html", active="dashboard")


# ============================================================
# DATA LOADING
# ============================================================

@app.route("/data-loading")
def data_loading():
    try:
        data = cached_data_summary()
        return render_template("data_loading.html", data=data, active="data-loading")
    except Exception as exc:
        return render_template("data_loading.html", data=None, error=str(exc), active="data-loading")


# ============================================================
# EDA (15 TASKS)
# ============================================================

@app.route("/eda")
def eda_page():
    try:
        results = cached_eda()
        return render_template(
            "eda.html",
            summary=results,
            charts=results.get("charts", []),
            active="eda"
        )
    except Exception as exc:
        return render_template("eda.html", summary=None, charts=[], error=str(exc), active="eda")


# ============================================================
# PREPROCESSING
# ============================================================

@app.route("/preprocessing")
def preprocessing_page():
    try:
        results = cached_preprocessing()
        return render_template("preprocessing.html", results=results, active="preprocessing")
    except Exception as exc:
        return render_template("preprocessing.html", results=None, error=str(exc), active="preprocessing")


# ============================================================
# MODELS HOME
# ============================================================

@app.route("/models")
def models_page():
    return render_template("models.html", active="models")


# ============================================================
# LINEAR REGRESSION
# ============================================================

@app.route("/models/linear-regression")
def linear_regression():
    try:
        results = cached_linear_regression()
        return render_template("model_result.html", results=results, model_name="Linear Regression", active="regression-linear")
    except Exception as exc:
        return render_template("model_result.html", results=None, model_name="Linear Regression", error=str(exc), active="regression-linear")


# ============================================================
# LOGISTIC REGRESSION
# ============================================================

@app.route("/models/logistic-regression")
def logistic_regression():
    try:
        results = cached_logistic_regression()
        return render_template("model_result.html", results=results, model_name="Logistic Regression", active="regression-logistic")
    except Exception as exc:
        return render_template("model_result.html", results=None, model_name="Logistic Regression", error=str(exc), active="regression-logistic")


# ============================================================
# REGULARISATION
# ============================================================

@app.route("/models/regularisation")
def regularisation():
    try:
        results = cached_regularisation()
        return render_template("model_result.html", results=results, model_name="Regularisation", active="regression-regularisation")
    except Exception as exc:
        return render_template("model_result.html", results=None, model_name="Regularisation", error=str(exc), active="regression-regularisation")


# ============================================================
# DECISION TREE
# ============================================================

@app.route("/models/decision-tree")
def decision_tree():
    try:
        results = cached_decision_tree()
        return render_template("model_result.html", results=results, model_name="Decision Tree", active="trees-decision")
    except Exception as exc:
        return render_template("model_result.html", results=None, model_name="Decision Tree", error=str(exc), active="trees-decision")


# ============================================================
# COST COMPLEXITY
# ============================================================

@app.route("/models/cost-complexity")
def cost_complexity():
    try:
        results = cached_cost_complexity()
        return render_template("model_result.html", results=results, model_name="Cost-Complexity Pruning", active="trees-cost")
    except Exception as exc:
        return render_template("model_result.html", results=None, model_name="Cost-Complexity Pruning", error=str(exc), active="trees-cost")


# ============================================================
# RANDOM FOREST
# ============================================================

@app.route("/models/random-forest")
def random_forest():
    try:
        results = cached_random_forest()
        return render_template("model_result.html", results=results, model_name="Random Forest", active="trees-forest")
    except Exception as exc:
        return render_template("model_result.html", results=None, model_name="Random Forest", error=str(exc), active="trees-forest")


# ============================================================
# ADABOOST
# ============================================================

@app.route("/models/ada-boost")
def ada_boost():
    try:
        results = cached_adaboost()
        return render_template("model_result.html", results=results, model_name="AdaBoost", active="trees-adaboost")
    except Exception as exc:
        return render_template("model_result.html", results=None, model_name="AdaBoost", error=str(exc), active="trees-adaboost")


# ============================================================
# GRADIENT BOOSTING
# ============================================================

@app.route("/models/gradient-boost")
def gradient_boost():
    try:
        results = cached_gradient_boosting()
        return render_template("model_result.html", results=results, model_name="Gradient Boosting", active="trees-gradient")
    except Exception as exc:
        return render_template("model_result.html", results=None, model_name="Gradient Boosting", error=str(exc), active="trees-gradient")


# ============================================================
# XGBOOST
# ============================================================

@app.route("/models/xg-boost")
def xg_boost():
    try:
        results = cached_xgboost()
        return render_template("model_result.html", results=results, model_name="XGBoost", active="trees-xgboost")
    except Exception as exc:
        return render_template("model_result.html", results=None, model_name="XGBoost", error=str(exc), active="trees-xgboost")


# ============================================================
# K-MEANS DROPDOWN ROUTES
# ============================================================

@app.route("/clustering/kmeans/manual")
def kmeans_manual():
    try:
        k = request.args.get("k", 3, type=int)
        results = cached_kmeans(n_clusters=k)
        return render_template("kmeans.html", mode="manual", results=results, selected_k=k, active="kmeans-manual")
    except Exception as exc:
        return render_template("kmeans.html", mode="manual", results=None, error=str(exc), active="kmeans-manual")


@app.route("/clustering/kmeans/elbow")
def kmeans_elbow():
    try:
        k = request.args.get("k", None, type=int)
        results = cached_k_calculation()
        return render_template("kmeans.html", mode="elbow", results=results, selected_k=k, active="kmeans-elbow")
    except Exception as exc:
        return render_template("kmeans.html", mode="elbow", results=None, error=str(exc), active="kmeans-elbow")


@app.route("/clustering/kmeans/silhouette")
def kmeans_silhouette():
    try:
        k = request.args.get("k", None, type=int)
        results = cached_k_calculation()
        return render_template("kmeans.html", mode="silhouette", results=results, selected_k=k, active="kmeans-silhouette")
    except Exception as exc:
        return render_template("kmeans.html", mode="silhouette", results=None, error=str(exc), active="kmeans-silhouette")


# Backward compatibility redirects
@app.route("/models/k-calculation")
def k_calculation():
    return redirect(url_for("kmeans_elbow"))


@app.route("/models/kmeans")
def kmeans():
    return redirect(url_for("kmeans_manual"))


@app.route("/models/kmeans-performance")
def kmeans_performance():
    return redirect(url_for("kmeans_silhouette"))


# ============================================================
# HIERARCHICAL CLUSTERING — 5 LINKAGE METHODS
# ============================================================

@app.route("/clustering/hierarchical")
def hierarchical_page():
    return redirect(url_for("hierarchical_ward"))


@app.route("/clustering/hierarchical/ward")
def hierarchical_ward():
    try:
        k = request.args.get("k", 3, type=int)
        results = cached_hierarchical(linkage_method="ward", n_clusters=k)
        return render_template("hierarchical.html", results=results, active="hierarchical-ward")
    except Exception as exc:
        return render_template("hierarchical.html", results=None, error=str(exc), active="hierarchical-ward")


@app.route("/clustering/hierarchical/complete")
def hierarchical_complete():
    try:
        k = request.args.get("k", 3, type=int)
        results = cached_hierarchical(linkage_method="complete", n_clusters=k)
        return render_template("hierarchical.html", results=results, active="hierarchical-complete")
    except Exception as exc:
        return render_template("hierarchical.html", results=None, error=str(exc), active="hierarchical-complete")


@app.route("/clustering/hierarchical/average")
def hierarchical_average():
    try:
        k = request.args.get("k", 3, type=int)
        results = cached_hierarchical(linkage_method="average", n_clusters=k)
        return render_template("hierarchical.html", results=results, active="hierarchical-average")
    except Exception as exc:
        return render_template("hierarchical.html", results=None, error=str(exc), active="hierarchical-average")


@app.route("/clustering/hierarchical/single")
def hierarchical_single():
    try:
        k = request.args.get("k", 3, type=int)
        results = cached_hierarchical(linkage_method="single", n_clusters=k)
        return render_template("hierarchical.html", results=results, active="hierarchical-single")
    except Exception as exc:
        return render_template("hierarchical.html", results=None, error=str(exc), active="hierarchical-single")


@app.route("/clustering/hierarchical/centroid")
def hierarchical_centroid():
    try:
        k = request.args.get("k", 3, type=int)
        results = cached_hierarchical(linkage_method="centroid", n_clusters=k)
        return render_template("hierarchical.html", results=results, active="hierarchical-centroid")
    except Exception as exc:
        return render_template("hierarchical.html", results=None, error=str(exc), active="hierarchical-centroid")


# ============================================================
# DBSCAN — PRESETS
# ============================================================

@app.route("/clustering/dbscan")
def dbscan_page():
    return redirect(url_for("dbscan_medium"))


@app.route("/clustering/dbscan/dense")
def dbscan_dense():
    try:
        results = cached_dbscan(preset="dense")
        return render_template("dbscan.html", results=results, active="dbscan-dense")
    except Exception as exc:
        return render_template("dbscan.html", results=None, error=str(exc), active="dbscan-dense")


@app.route("/clustering/dbscan/medium")
def dbscan_medium():
    try:
        results = cached_dbscan(preset="medium")
        return render_template("dbscan.html", results=results, active="dbscan-medium")
    except Exception as exc:
        return render_template("dbscan.html", results=None, error=str(exc), active="dbscan-medium")


@app.route("/clustering/dbscan/sparse")
def dbscan_sparse():
    try:
        results = cached_dbscan(preset="sparse")
        return render_template("dbscan.html", results=results, active="dbscan-sparse")
    except Exception as exc:
        return render_template("dbscan.html", results=None, error=str(exc), active="dbscan-sparse")


# ============================================================
# ANOMALY DETECTION — 4 METHODS
# ============================================================

@app.route("/clustering/anomaly")
def anomaly_page():
    return redirect(url_for("anomaly_isolation_forest"))


@app.route("/clustering/anomaly/isolation-forest")
def anomaly_isolation_forest():
    try:
        results = cached_anomaly(method="isolation-forest")
        return render_template("anomaly.html", results=results, active="anomaly-isolation-forest")
    except Exception as exc:
        return render_template("anomaly.html", results=None, error=str(exc), active="anomaly-isolation-forest")


@app.route("/clustering/anomaly/one-class-svm")
def anomaly_one_class_svm():
    try:
        results = cached_anomaly(method="one-class-svm")
        return render_template("anomaly.html", results=results, active="anomaly-one-class-svm")
    except Exception as exc:
        return render_template("anomaly.html", results=None, error=str(exc), active="anomaly-one-class-svm")


@app.route("/clustering/anomaly/reconstruction")
def anomaly_reconstruction():
    try:
        results = cached_anomaly(method="reconstruction")
        return render_template("anomaly.html", results=results, active="anomaly-reconstruction")
    except Exception as exc:
        return render_template("anomaly.html", results=None, error=str(exc), active="anomaly-reconstruction")


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":
    print("Warming up ML caches...")
    try:
        cached_data_summary()
        cached_eda()
    except Exception as e:
        print(f"Cache warm-up notice: {e}")

    app.run(debug=True, host="127.0.0.1", port=5000)
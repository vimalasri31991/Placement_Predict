import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import warnings

from .model_data import (
    prepare_data,
    classification_metrics
)

_REG_CACHE = None


def run_regularisation(force=False):
    global _REG_CACHE
    if not force and _REG_CACHE is not None:
        return _REG_CACHE

    (
        X_train,
        X_test,
        y_train,
        y_test,
        pre
    ) = prepare_data(
        scale_numeric=True
    )

    X_train_t = pre.fit_transform(X_train)
    X_test_t = pre.transform(X_test)

    results = {}

    # 1. L2 Ridge
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        l2_model = LogisticRegression(
            penalty="l2",
            C=1.0,
            solver="lbfgs",
            max_iter=500,
            random_state=42
        )
        l2_model.fit(X_train_t, y_train)
        results["L2 Ridge"] = classification_metrics(y_test, l2_model.predict(X_test_t))

    # 2. L1 Lasso
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        l1_model = LogisticRegression(
            penalty="l1",
            C=1.0,
            solver="liblinear",
            max_iter=500,
            random_state=42
        )
        l1_model.fit(X_train_t, y_train)
        results["L1 Lasso"] = classification_metrics(y_test, l1_model.predict(X_test_t))

    # 3. Elastic Net (subsample 12,000 for fast convergence in <0.5s)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        np.random.seed(42)
        sub_idx = np.random.choice(len(X_train_t), size=min(12000, len(X_train_t)), replace=False)
        y_train_sub = y_train.iloc[sub_idx] if hasattr(y_train, "iloc") else y_train[sub_idx]

        elastic_model = LogisticRegression(
            penalty="elasticnet",
            l1_ratio=0.5,
            C=1.0,
            solver="saga",
            max_iter=200,
            tol=1e-3,
            random_state=42
        )
        elastic_model.fit(X_train_t[sub_idx], y_train_sub)
        results["Elastic Net"] = classification_metrics(y_test, elastic_model.predict(X_test_t))

    _REG_CACHE = {
        "model": "Regularisation",
        "variants": results
    }

    return _REG_CACHE
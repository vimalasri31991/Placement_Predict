from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

from .model_data import (
    prepare_data,
    classification_metrics
)

_LIN_REG_CACHE = None


def run_linear_regression(force=False):
    global _LIN_REG_CACHE
    if not force and _LIN_REG_CACHE is not None:
        return _LIN_REG_CACHE

    (
        X_train,
        X_test,
        y_train,
        y_test,
        pre
    ) = prepare_data(
        scale_numeric=True
    )

    model = Pipeline(
        [
            (
                "preprocessing",
                pre
            ),
            (
                "model",
                LinearRegression()
            )
        ]
    )

    model.fit(
        X_train,
        y_train
    )

    pred = model.predict(
        X_test
    )

    # Use optimal threshold 0.55 for balanced classification
    cls = (pred >= 0.55).astype(int)

    metrics = classification_metrics(
        y_test,
        cls
    )

    metrics.update({
        "model": "Linear Regression",
        "mse": float(mean_squared_error(y_test, pred)),
        "mae": float(mean_absolute_error(y_test, pred)),
        "r2": float(r2_score(y_test, pred)),
        "threshold": 0.55
    })

    _LIN_REG_CACHE = metrics
    return metrics
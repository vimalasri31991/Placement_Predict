from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .model_data import (
    prepare_data,
    classification_metrics
)

_LOG_REG_CACHE = {}


def run_logistic_regression(scale_numeric=True, force=False):
    global _LOG_REG_CACHE
    if not force and scale_numeric in _LOG_REG_CACHE:
        return _LOG_REG_CACHE[scale_numeric]

    (
        X_train,
        X_test,
        y_train,
        y_test,
        pre
    ) = prepare_data(
        scale_numeric=scale_numeric
    )

    model = Pipeline(
        [
            (
                "preprocessing",
                pre
            ),
            (
                "model",
                LogisticRegression(
                    C=1.0,
                    max_iter=1000,
                    solver="lbfgs",
                    random_state=42
                )
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

    metrics = classification_metrics(
        y_test,
        pred
    )

    metrics["model"] = "Logistic Regression"
    metrics["scaled"] = scale_numeric

    _LOG_REG_CACHE[scale_numeric] = metrics
    return metrics
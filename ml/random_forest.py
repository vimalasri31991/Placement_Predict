from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from .model_data import (
    prepare_data,
    classification_metrics
)

_RF_CACHE = None


def run_random_forest(force=False):
    global _RF_CACHE
    if not force and _RF_CACHE is not None:
        return _RF_CACHE

    (
        X_train,
        X_test,
        y_train,
        y_test,
        pre
    ) = prepare_data(
        scale_numeric=False
    )

    model = Pipeline(
        [
            (
                "preprocessing",
                pre
            ),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=100,
                    max_depth=15,
                    min_samples_split=8,
                    min_samples_leaf=3,
                    random_state=42,
                    n_jobs=-1
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

    metrics.update({
        "model": "Random Forest",
        "n_estimators": 100,
        "max_depth": 15
    })

    _RF_CACHE = metrics
    return metrics
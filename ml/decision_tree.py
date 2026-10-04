from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline

from .model_data import (
    prepare_data,
    classification_metrics
)

_DT_CACHE = None


def run_decision_tree(force=False):
    global _DT_CACHE
    if not force and _DT_CACHE is not None:
        return _DT_CACHE

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
                DecisionTreeClassifier(
                    max_depth=12,
                    min_samples_split=10,
                    min_samples_leaf=5,
                    criterion="gini",
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

    metrics.update({
        "model": "Decision Tree",
        "max_depth": 12,
        "min_samples_split": 10,
        "min_samples_leaf": 5
    })

    _DT_CACHE = metrics
    return metrics
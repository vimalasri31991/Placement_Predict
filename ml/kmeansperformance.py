from sklearn.cluster import KMeans

from sklearn.metrics import (
    silhouette_score,
    adjusted_rand_score
)

from .model_data import get_model_data


def evaluate_kmeans(
        k=3
):

    (
        X_train,
        X_test,
        y_train,
        y_test,
        _
    ) = get_model_data(
        scale_numeric=True
    )

    model = KMeans(

        n_clusters=k,

        random_state=42,

        n_init=10
    )

    labels = model.fit_predict(
        X_test
    )

    return {

        "k":
            k,

        "inertia":
            float(
                model.inertia_
            ),

        "silhouette":
            float(
                silhouette_score(
                    X_test,
                    labels
                )
            ),

        "adjusted_rand_index_vs_placement":
            float(
                adjusted_rand_score(
                    y_test,
                    labels
                )
            )
    }
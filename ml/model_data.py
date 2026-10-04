import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder
)

from sklearn.impute import SimpleImputer

from sklearn.pipeline import Pipeline

from .data_loader import load_data


TARGET = "PlacementStatus"


DROP_COLUMNS = [
    "StudentID",
    "Salary Package",
    "IsAnomaly"
]


def _encoder():

    try:

        return OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        )

    except TypeError:

        return OneHotEncoder(
            handle_unknown="ignore",
            sparse=False
        )


def prepare_data(
        scale_numeric=True,
        test_size=0.20
):

    df = load_data().copy()

    # Remove duplicate rows
    df = df.drop_duplicates()

    # Remove columns that should not be used
    df = df.drop(
        columns=[
            c for c in DROP_COLUMNS
            if c in df.columns
        ]
    )

    # Convert target
    df[TARGET] = pd.to_numeric(
        df[TARGET],
        errors="coerce"
    )

    # Remove rows where target is missing
    df = df.dropna(
        subset=[TARGET]
    )

    X = df.drop(
        columns=TARGET
    )

    y = df[TARGET].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(

        X,
        y,

        test_size=test_size,

        random_state=42,

        stratify=y
    )

    numeric = (
        X_train
        .select_dtypes(include="number")
        .columns
        .tolist()
    )

    categorical = (
        X_train
        .select_dtypes(
            include=[
                "object",
                "category",
                "bool"
            ]
        )
        .columns
        .tolist()
    )

    transformers = []

    # Numeric preprocessing
    if numeric:

        steps = [

            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            )
        ]

        if scale_numeric:

            steps.append(
                (
                    "scaler",
                    StandardScaler()
                )
            )

        transformers.append(

            (
                "numeric",
                Pipeline(steps),
                numeric
            )
        )

    # Categorical preprocessing
    if categorical:

        transformers.append(

            (
                "categorical",

                Pipeline(
                    [
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="most_frequent"
                            )
                        ),

                        (
                            "encoder",
                            _encoder()
                        )
                    ]
                ),

                categorical
            )
        )

    preprocessor = ColumnTransformer(

        transformers=transformers,

        remainder="drop"
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        preprocessor
    )


def get_model_data(
        scale_numeric=True
):

    (
        X_train,
        X_test,
        y_train,
        y_test,
        preprocessor
    ) = prepare_data(
        scale_numeric
    )

    X_train_t = preprocessor.fit_transform(
        X_train
    )

    X_test_t = preprocessor.transform(
        X_test
    )

    try:

        names = (
            preprocessor
            .get_feature_names_out()
            .tolist()
        )

    except Exception:

        names = [
            f"feature_{i}"
            for i in range(
                X_train_t.shape[1]
            )
        ]

    return (
        X_train_t,
        X_test_t,
        y_train,
        y_test,
        names
    )


def classification_metrics(
        y_true,
        y_pred
):

    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        confusion_matrix
    )

    return {

        "accuracy": float(
            accuracy_score(
                y_true,
                y_pred
            )
        ),

        "precision": float(
            precision_score(
                y_true,
                y_pred,
                zero_division=0
            )
        ),

        "recall": float(
            recall_score(
                y_true,
                y_pred,
                zero_division=0
            )
        ),

        "f1": float(
            f1_score(
                y_true,
                y_pred,
                zero_division=0
            )
        ),

        "confusion_matrix":
            confusion_matrix(
                y_true,
                y_pred
            ).tolist()
    }
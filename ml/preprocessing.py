from pathlib import Path

import pandas as pd

from sklearn.model_selection import train_test_split

from .data_loader import load_data

from .class_encoding import (
    encode_ordinal_columns
)

from .standard_scaling import (
    get_standard_scaling_preview
)

from .min_max_scaling import (
    get_minmax_scaling_preview
)

from .one_hot_encoding import (
    one_hot_encode
)

from .outlier_fix import (
    detect_outliers_iqr,
    clip_outliers_iqr
)


BASE_DIR = Path(__file__).resolve().parent.parent

PREPROCESSED_PATH = (
    BASE_DIR /
    "preprocessed_data.csv"
)


TARGET = "PlacementStatus"


DROP_FOR_MODEL = [
    "StudentID",
    "Salary Package",
    "IsAnomaly"
]


def clean_dataframe(df):

    out = df.copy()

    # Remove duplicates
    out = out.drop_duplicates()

    # Numeric missing values
    numeric_columns = (
        out
        .select_dtypes(include="number")
        .columns
    )

    for col in numeric_columns:

        out[col] = out[col].fillna(
            out[col].median()
        )

    # Categorical missing values
    categorical_columns = (
        out
        .select_dtypes(
            include=[
                "object",
                "category",
                "bool"
            ]
        )
        .columns
    )

    for col in categorical_columns:

        mode = out[col].mode(
            dropna=True
        )

        if not mode.empty:

            out[col] = out[col].fillna(
                mode.iloc[0]
            )

        else:

            out[col] = out[col].fillna(
                "Unknown"
            )

    return out


def create_preprocessed_file():

    df = load_data()

    df = clean_dataframe(
        df
    )

    # Ordinal encoding
    df, _ = encode_ordinal_columns(
        df
    )

    # Detect numerical columns
    numeric_columns = (
        df
        .select_dtypes(include="number")
        .columns
        .tolist()
    )

    numeric_columns = [
        c for c in numeric_columns
        if c != TARGET
    ]

    # Fix outliers
    df, _ = clip_outliers_iqr(
        df,
        numeric_columns
    )

    df.to_csv(
        PREPROCESSED_PATH,
        index=False
    )

    return df


def run_preprocessing():

    df = load_data().copy()

    before = len(df)

    clean = clean_dataframe(
        df
    )

    after = len(clean)

    train_df, test_df = train_test_split(

        clean,

        test_size=0.30,

        random_state=42,

        stratify=clean[TARGET]
    )

    numeric = (
        clean
        .select_dtypes(include="number")
        .columns
        .tolist()
    )

    nominal = (
        clean
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

    ordinal = [
        c for c in [
            "CollegeTier",
            "CGPA_Tier"
        ]
        if c in clean.columns
    ]

    preview_df = clean.drop(
        columns=[
            c for c in DROP_FOR_MODEL
            if c in clean.columns
        ],
        errors="ignore"
    )

    preview_numeric = [

        c for c in
        preview_df
        .select_dtypes(include="number")
        .columns

        if c != TARGET
    ]

    # Standard Scaling
    standard_preview, _ = (
        get_standard_scaling_preview(
            preview_df,
            preview_numeric[:8]
        )
    )

    # Min-Max Scaling
    minmax_preview, _ = (
        get_minmax_scaling_preview(
            preview_df,
            preview_numeric[:8]
        )
    )

    # One Hot Encoding
    ohe_source = (
        clean[nominal]
        .head(10)
        .copy()
        if nominal
        else pd.DataFrame()
    )

    (
        ohe_preview,
        encoded_columns,
        _
    ) = one_hot_encode(
        ohe_source,
        nominal
    )

    # Ordinal Encoding
    (
        ordinal_preview,
        ordinal_encoded_columns
    ) = encode_ordinal_columns(
        clean.head(10)
    )

    # Outlier Detection
    outliers = detect_outliers_iqr(
        clean,
        preview_numeric
    )

    # Outlier Clipping
    _, clipping = clip_outliers_iqr(
        clean,
        preview_numeric
    )

    # Save processed dataset
    processed = create_preprocessed_file()

    return {

        "rows_before":
            before,

        "rows_after_duplicates":
            after,

        "duplicate_count":
            before - after,

        "train_rows":
            len(train_df),

        "test_rows":
            len(test_df),

        "train_percentage":
            70,

        "test_percentage":
            30,

        "numeric_columns":
            numeric,

        "nominal_columns":
            nominal,

        "ordinal_columns":
            ordinal,

        "standard_preview":
            standard_preview,

        "minmax_preview":
            minmax_preview,

        "ohe_preview":
            ohe_preview
            .to_dict(
                orient="records"
            ),

        "encoded_column_count":
            len(encoded_columns),

        "encoded_columns":
            encoded_columns,

        "ordinal_encoded_columns":
            ordinal_encoded_columns,

        "ordinal_preview":
            ordinal_preview
            .to_dict(
                orient="records"
            ),

        "outliers":
            outliers,

        "clipping":
            clipping,

        "processed_rows":
            len(processed),

        "target":
            TARGET
    }
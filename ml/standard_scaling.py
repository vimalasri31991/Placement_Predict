import pandas as pd

from sklearn.preprocessing import StandardScaler


def standard_scale(df, columns=None):

    out = df.copy()

    if columns is None:

        columns = (
            out
            .select_dtypes(include="number")
            .columns
            .tolist()
        )

    columns = [
        c for c in columns
        if c in out.columns
    ]

    scaler = StandardScaler()

    if columns:

        out[columns] = scaler.fit_transform(
            out[columns].fillna(
                out[columns].median()
            )
        )

    return out, scaler


def get_standard_scaling_preview(
        df,
        columns=None,
        n=5
):

    out, scaler = standard_scale(
        df,
        columns
    )

    if columns is None:

        columns = (
            df
            .select_dtypes(include="number")
            .columns
            .tolist()
        )

    return (
        out[columns]
        .head(n)
        .round(4)
        .to_dict(orient="records"),
        scaler
    )
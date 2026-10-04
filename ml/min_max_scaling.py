import pandas as pd

from sklearn.preprocessing import MinMaxScaler


def min_max_scale(df, columns=None):

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

    scaler = MinMaxScaler()

    if columns:

        out[columns] = scaler.fit_transform(
            out[columns].fillna(
                out[columns].median()
            )
        )

    return out, scaler


def get_minmax_scaling_preview(
        df,
        columns=None,
        n=5
):

    out, scaler = min_max_scale(
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
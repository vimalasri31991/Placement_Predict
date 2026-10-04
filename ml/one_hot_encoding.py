import pandas as pd

from sklearn.preprocessing import OneHotEncoder


def make_encoder():

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


def one_hot_encode(
        df,
        columns=None
):

    out = df.copy()

    if columns is None:

        columns = (
            out
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

    columns = [
        c for c in columns
        if c in out.columns
    ]

    if not columns:

        return out, [], None

    encoder = make_encoder()

    values = encoder.fit_transform(
        out[columns]
        .fillna("Missing")
        .astype(str)
    )

    names = (
        encoder
        .get_feature_names_out(columns)
        .tolist()
    )

    encoded = pd.DataFrame(
        values,
        columns=names,
        index=out.index
    )

    out = pd.concat(
        [
            out.drop(columns=columns),
            encoded
        ],
        axis=1
    )

    return out, names, encoder
import pandas as pd


def detect_outliers_iqr(
        df,
        columns=None
):

    if columns is None:

        columns = (
            df
            .select_dtypes(include="number")
            .columns
            .tolist()
        )

    result = []

    for col in columns:

        if col not in df.columns:
            continue

        s = pd.to_numeric(
            df[col],
            errors="coerce"
        ).dropna()

        if s.empty:
            continue

        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)

        iqr = q3 - q1

        low = q1 - 1.5 * iqr
        high = q3 + 1.5 * iqr

        count = int(
            (
                (s < low) |
                (s > high)
            ).sum()
        )

        result.append({

            "column": col,

            "q1": round(
                float(q1),
                3
            ),

            "q3": round(
                float(q3),
                3
            ),

            "lower": round(
                float(low),
                3
            ),

            "upper": round(
                float(high),
                3
            ),

            "outliers": count
        })

    return result


def clip_outliers_iqr(
        df,
        columns=None
):

    out = df.copy()

    if columns is None:

        columns = (
            out
            .select_dtypes(include="number")
            .columns
            .tolist()
        )

    clipping = []

    for col in columns:

        if col not in out.columns:
            continue

        s = pd.to_numeric(
            out[col],
            errors="coerce"
        )

        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)

        iqr = q3 - q1

        low = q1 - 1.5 * iqr
        high = q3 + 1.5 * iqr

        before = int(
            (
                (s < low) |
                (s > high)
            ).sum()
        )

        out[col] = s.clip(
            low,
            high
        )

        clipping.append({

            "column": col,

            "clipped": before,

            "lower": round(
                float(low),
                3
            ),

            "upper": round(
                float(high),
                3
            )
        })

    return out, clipping
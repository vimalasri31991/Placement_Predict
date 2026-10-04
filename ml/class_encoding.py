import pandas as pd


TARGET = "PlacementStatus"


ORDINAL_COLUMNS = [
    "CollegeTier",
    "CGPA_Tier"
]


ORDINAL_MAPPINGS = {
    "CollegeTier": {
        "Tier3": 0,
        "Tier2": 1,
        "Tier1": 2
    },

    "CGPA_Tier": {
        "Low": 0,
        "Medium": 1,
        "High": 2
    }
}


def encode_classes(df: pd.DataFrame) -> pd.DataFrame:

    out = df.copy()

    for col, mapping in ORDINAL_MAPPINGS.items():

        if col in out.columns:

            out[col] = (
                out[col]
                .astype(str)
                .str.strip()
                .map(mapping)
            )

    if TARGET in out.columns:

        out[TARGET] = pd.to_numeric(
            out[TARGET],
            errors="coerce"
        )

    return out


def encode_ordinal_columns(df: pd.DataFrame):

    out = df.copy()

    encoded = []

    for col, mapping in ORDINAL_MAPPINGS.items():

        if col in out.columns:

            out[col] = (
                out[col]
                .astype(str)
                .str.strip()
                .map(mapping)
            )

            encoded.append(col)

    return out, encoded
from pathlib import Path
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "placement_predict_50k Dataset (3)(in).csv"
PREPROCESSED_PATH = BASE_DIR / "preprocessed_data.csv"


def load_data():
    """
    Load the original placement dataset.
    """

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {DATA_PATH}"
        )

    return pd.read_csv(DATA_PATH)


def load_preprocessed_data():
    """
    Load preprocessed data.
    If the file does not exist, create it first.
    """

    if not PREPROCESSED_PATH.exists():
        from .preprocessing import create_preprocessed_file
        create_preprocessed_file()

    return pd.read_csv(PREPROCESSED_PATH)


def get_data_summary():
    """
    Return basic information about the dataset.
    """

    df = load_data()

    missing = df.isnull().sum()

    summary = []

    for col in df.columns:
        summary.append({
            "column": col,
            "dtype": str(df[col].dtype),
            "missing": int(missing[col]),
            "unique": int(df[col].nunique(dropna=True))
        })

    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "column_names": list(df.columns),
        "missing_total": int(df.isnull().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "preview": df.head(10).to_dict(orient="records"),
        "summary": summary
    }
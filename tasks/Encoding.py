"""Task 2: categorical encoding and missing-value handling."""
import logging
import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_data, normalize_strings, OUTPUT_DIR, TARGET

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Encoding")

BINARY_YN = {"yes": 1, "no": 0}


def run():
    path = os.path.join(OUTPUT_DIR, "outlier_skewness_data.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(
        "outlier_skewness_data.pkl not found. "
        "Run OutlierSkewness first."
    )
    df = pd.read_pickle(path) if os.path.exists(path) else normalize_strings(load_data()).drop_duplicates().reset_index(drop=True)

    # Missing categorical values -> mode; numeric values -> median.
    for col in df.columns:
        if df[col].isna().any():
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].median())
            else:
                mode = df[col].mode(dropna=True)
                if not mode.empty:
                    df[col] = df[col].fillna(mode.iloc[0])

    if TARGET not in df.columns:
        raise ValueError("Expected target column 'y' not found.")

    df[TARGET] = df[TARGET].map(BINARY_YN).astype(int)

    # Preserve numeric columns and one-hot encode remaining categoricals.
    cat_cols = [c for c in df.select_dtypes(include=["object", "category"]).columns if c != TARGET]
    encoded = pd.get_dummies(df, columns=cat_cols, dtype=int)

    # Make sure any bool columns are numeric.
    for c in encoded.columns:
        if encoded[c].dtype == bool:
            encoded[c] = encoded[c].astype(int)

    out = os.path.join(OUTPUT_DIR, "encoded_data.pkl")
    encoded.to_pickle(out)
    logger.info("Encoded categorical columns: %d; resulting shape=%s", len(cat_cols), encoded.shape)
    return encoded


if __name__ == "__main__":
    run()

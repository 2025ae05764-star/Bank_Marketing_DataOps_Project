"""Task 3: standardization of numeric features, matching the notebook's model preparation."""
import logging
import os
import sys
import pandas as pd
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import OUTPUT_DIR, TARGET

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Normalization")


def run():
    source = os.path.join(OUTPUT_DIR, "encoded_data.pkl")
    if not os.path.exists(source):
        raise FileNotFoundError("encoded_data.pkl not found. Run Encoding first.")
    df = pd.read_pickle(source)
    feature_cols = [c for c in df.columns if c != TARGET]
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])

    # Drop almost-constant columns, as done in the notebook.
    variances = df[feature_cols].var()
    low_var = variances[variances < 0.01].index.tolist()
    df = df.drop(columns=low_var)
    out = os.path.join(OUTPUT_DIR, "normalized_data.pkl")
    df.to_pickle(out)
    pd.Series(low_var, name="low_variance_column").to_csv(os.path.join(OUTPUT_DIR, "low_variance_columns.csv"), index=False)
    logger.info("Standardized %d features; dropped %d low-variance features; shape=%s", len(feature_cols), len(low_var), df.shape)
    return df


if __name__ == "__main__":
    run()

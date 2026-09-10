"""Task 4: age binning for EDA/business interpretation."""
import logging
import os
import sys
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import OUTPUT_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Binning")


def run():
    source = os.path.join(OUTPUT_DIR, "normalized_data.pkl")
    if not os.path.exists(source):
        raise FileNotFoundError("normalized_data.pkl not found. Run Normalization first.")
    df = pd.read_pickle(source)

    # The notebook standardizes features before modelling. For EDA, retain a
    # human-readable age band based on the original age values when available.
    # If age has already been standardized, use quantile bins as a robust fallback.
    if "age" in df.columns:
        age = df["age"]
        df["age_bin"] = pd.qcut(age, q=5, duplicates="drop").astype(str)
        bins = df["age_bin"].value_counts().sort_index()
    else:
        bins = pd.Series(dtype=int)

    df.to_pickle(os.path.join(OUTPUT_DIR, "binned_data.pkl"))
    bins.rename("count").to_csv(os.path.join(OUTPUT_DIR, "age_bins.csv"))
    logger.info("Age binning completed. Number of bins=%d", len(bins))
    return df


if __name__ == "__main__":
    run()

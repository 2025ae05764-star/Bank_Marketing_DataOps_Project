"""Task 1: data ingestion, summary statistics, missing values and dtypes."""
import logging
import os
import sys
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_data, normalize_strings, OUTPUT_DIR, TARGET

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("BasicStats")


def run():
    df = normalize_strings(load_data())
    duplicates = int(df.duplicated().sum())
    missing = df.isna().sum()
    numeric_summary = df.describe(include="number").T
    dtypes = df.dtypes.astype(str)

    logger.info("Ingested dataset: rows=%d, columns=%d", *df.shape)
    logger.info("Target '%s' distribution:\n%s", TARGET, df[TARGET].value_counts(dropna=False).to_string())
    logger.info("Duplicate rows: %d", duplicates)
    logger.info("Missing values:\n%s", missing.to_string())
    logger.info("Data types:\n%s", dtypes.to_string())

    numeric_summary.to_csv(os.path.join(OUTPUT_DIR, "summary_statistics.csv"))
    missing.rename("missing_count").to_csv(os.path.join(OUTPUT_DIR, "missing_values.csv"))
    dtypes.rename("dtype").to_csv(os.path.join(OUTPUT_DIR, "data_types.csv"))

    cleaned = df.drop_duplicates().reset_index(drop=True)
    cleaned.to_pickle(os.path.join(OUTPUT_DIR, "ingested_cleaned.pkl"))
    return {"rows": len(df), "columns": len(df.columns), "duplicates": duplicates}


if __name__ == "__main__":
    run()

"""Task 5: Pearson correlation matrix and heatmap."""
import logging
import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_data, normalize_strings, OUTPUT_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("CorrelationCoefficient")


def run():
    df = normalize_strings(load_data())
    numeric = df.select_dtypes(include="number")
    corr = numeric.corr(method="pearson")
    corr.to_csv(os.path.join(OUTPUT_DIR, "correlation_matrix.csv"))
    plt.figure(figsize=(12, 9))
    sns.heatmap(corr, cmap="coolwarm", center=0)
    plt.title("Bank Marketing — Numeric Feature Correlation Matrix")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "correlation_heatmap.png")
    plt.savefig(path, dpi=150)
    plt.close()
    logger.info("Correlation matrix calculated for %d numeric features; saved %s", len(numeric.columns), path)
    return corr


if __name__ == "__main__":
    run()

"""Task 7: univariate and bivariate EDA plots."""
import logging
import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_data, normalize_strings, OUTPUT_DIR, TARGET

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Visualization")


def run():
    df = normalize_strings(load_data())

    plt.figure(figsize=(7, 5))
    sns.countplot(data=df, x=TARGET)
    plt.title("Target Distribution — Term Deposit Subscription")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "target_distribution.png"), dpi=150)
    plt.close()

    numeric = df.select_dtypes(include="number")
    numeric.hist(figsize=(14, 10), bins=25)
    plt.suptitle("Numeric Feature Histograms")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "numeric_histograms.png"), dpi=150)
    plt.close("all")

    plt.figure(figsize=(14, 7))
    numeric.boxplot(rot=45)
    plt.title("Numeric Feature Boxplots")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "numeric_boxplots.png"), dpi=150)
    plt.close()

    if "age" in df.columns and "duration" in df.columns:
        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=df.sample(min(len(df), 10000), random_state=1), x="age", y="duration", hue=TARGET, alpha=0.5)
        plt.title("Age vs Contact Duration")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "scatter_age_duration.png"), dpi=150)
        plt.close()

    if "job" in df.columns:
        plt.figure(figsize=(12, 6))
        sns.countplot(data=df, y="job", hue=TARGET)
        plt.title("Job Category vs Subscription Outcome")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "job_vs_target.png"), dpi=150)
        plt.close()

    logger.info("EDA visualizations saved to %s", OUTPUT_DIR)
    return {"output_dir": OUTPUT_DIR}


if __name__ == "__main__":
    run()

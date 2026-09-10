"""Task 6: Pearson correlations between selected numeric variables."""
import logging
import os
import sys
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_data, normalize_strings

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("PearsonCorrelation")


def run():
    df = normalize_strings(load_data())
    pairs = [("age", "duration"), ("campaign", "duration"), ("age", "campaign")]
    results = {}
    for a, b in pairs:
        if a not in df.columns or b not in df.columns:
            continue
        r, p = stats.pearsonr(df[a], df[b])
        results[f"{a}_vs_{b}"] = {"r": float(r), "p_value": float(p)}
        logger.info("Pearson r(%s,%s)=%.4f, p=%.4g", a, b, r, p)
    return results


if __name__ == "__main__":
    run()

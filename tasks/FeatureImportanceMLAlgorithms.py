"""Task 8: Gradient Boosting feature importance, matching the supplied notebook."""
import logging
import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import OUTPUT_DIR, TARGET

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("FeatureImportanceMLAlgorithms")


def run():
    source = os.path.join(OUTPUT_DIR, "binned_data.pkl")
    if not os.path.exists(source):
        raise FileNotFoundError("binned_data.pkl not found. Run the preprocessing tasks first.")
    df = pd.read_pickle(source)
    df = df.drop(columns=["age_bin"], errors="ignore")

    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

    train_x, test_x, train_y, test_y = train_test_split(
        X, y, test_size=0.2, random_state=0, stratify=y
    )

    # Same primary algorithm and parameter search space family as the notebook.
    model = GradientBoostingClassifier(
        n_estimators=500,
        max_depth=7,
        learning_rate=0.01,
        subsample=1.0,
        random_state=1,
    )
    model.fit(train_x, train_y)
    pred = model.predict(test_x)
    prob = model.predict_proba(test_x)[:, 1]

    metrics = {
        "accuracy": accuracy_score(test_y, pred),
        "precision_weighted": precision_score(test_y, pred, average="weighted", zero_division=0),
        "recall_weighted": recall_score(test_y, pred, average="weighted", zero_division=0),
        "f1_weighted": f1_score(test_y, pred, average="weighted", zero_division=0),
        "auc_roc": roc_auc_score(test_y, prob),
    }
    pd.Series(metrics).to_csv(os.path.join(OUTPUT_DIR, "gradient_boosting_metrics.csv"), header=["value"])
    logger.info("Gradient Boosting metrics: %s", metrics)

    importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    importances.to_csv(os.path.join(OUTPUT_DIR, "gb_feature_importance.csv"), header=["importance"])

    plt.figure(figsize=(10, 8))
    importances.head(25).sort_values().plot(kind="barh")
    plt.title("Gradient Boosting — Top 25 Feature Importances")
    plt.xlabel("Importance")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "gb_feature_importance.png")
    plt.savefig(path, dpi=150)
    plt.close()

    logger.info("Top feature: %s", importances.index[0])
    return {"top_feature": importances.index[0], "metrics": metrics}


if __name__ == "__main__":
    run()

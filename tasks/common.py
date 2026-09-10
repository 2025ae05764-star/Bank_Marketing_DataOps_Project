import os
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(HERE)
DATA_PATH = os.path.join(PROJECT_DIR, "data", "bank-additional-full.xlsx")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "output")
TARGET = "y"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found: {path}. Place the exact bank-additional-full.xlsx used by the ML Lab notebook in data/."
        )
    return pd.read_excel(path)


def normalize_strings(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.select_dtypes(include=["object"]).columns:
        out[col] = out[col].map(lambda x: x.strip().lower() if isinstance(x, str) else x)
    return out

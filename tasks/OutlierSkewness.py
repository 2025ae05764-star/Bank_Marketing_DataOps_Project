import os
import logging
import numpy as np
import pandas as pd

from common import OUTPUT_DIR, TARGET, DATA_PATH

logger = logging.getLogger(__name__)


def handle_outliers_remove(
    df: pd.DataFrame,
    skew_threshold=1.0,
    winsor_limits=(0.01, 0.01),
    fill_with="median"
):
    """
    Detect and remove outlier rows.

    Logic follows the ML Lab notebook:
    - If |skew| <= 1.0, use IQR method.
    - If |skew| > 1.0, use 1%/99% percentile cutoffs.
    - Remove complete rows containing outliers.
    """

    df_clean = df.copy()

    # Only numeric columns are checked for outliers
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns

    summary_records = []
    total_outlier_rows = set()

    for col in numeric_cols:

        skew_val = df_clean[col].skew()

        # Low/moderate skew -> IQR method
        if abs(skew_val) <= skew_threshold:

            Q1, Q3 = df_clean[col].quantile([0.25, 0.75])
            IQR = Q3 - Q1

            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR

            method = "IQR"

        # High skew -> percentile method
        else:

            lower = df_clean[col].quantile(winsor_limits[0])
            upper = df_clean[col].quantile(
                1 - winsor_limits[1]
            )

            method = "Percentile Cutoff"

        # Identify outlier rows
        mask = (
            (df_clean[col] < lower) |
            (df_clean[col] > upper)
        )

        num_outliers = int(mask.sum())

        outlier_indices = df_clean[mask].index

        # Keep a unique set of rows
        total_outlier_rows.update(outlier_indices)

        summary_records.append({
            "Column": col,
            "Skew": round(skew_val, 3),
            "OutliersDetected": num_outliers,
            "Method": method,
            "LowerBound": lower,
            "UpperBound": upper
        })

    # Remove all rows identified as outliers
    df_clean = (
        df_clean
        .drop(index=list(total_outlier_rows))
        .reset_index(drop=True)
    )

    # Handle remaining numeric missing values
    if fill_with == "median":
        df_clean[numeric_cols] = (
            df_clean[numeric_cols]
            .fillna(df_clean[numeric_cols].median())
        )

    elif fill_with == "mean":
        df_clean[numeric_cols] = (
            df_clean[numeric_cols]
            .fillna(df_clean[numeric_cols].mean())
        )

    summary_df = pd.DataFrame(summary_records)

    rows_removed = len(total_outlier_rows)
    percentage_removed = (
        100 * rows_removed / len(df)
        if len(df) > 0
        else 0
    )

    logger.info(
        "Outlier detection completed: %d rows removed (%.2f%%)",
        rows_removed,
        percentage_removed
    )

    logger.info(
        "Outlier summary:\n%s",
        summary_df.to_string(index=False)
    )

    return df_clean, summary_df


def auto_fix_skewness(
    df: pd.DataFrame,
    skew_threshold=0.5
):
    """
    Correct skewness following the ML Lab notebook.

    - Skip binary/one-hot encoded columns.
    - |skew| > 0.5 triggers transformation.
    - skew > 1    -> log1p
    - skew < -1   -> square
    - otherwise   -> sqrt
    """

    df_transformed = df.copy()

    numeric_cols = (
        df_transformed
        .select_dtypes(include=[np.number])
        .columns
    )

    # Only continuous numeric columns
    continuous_cols = [
        col
        for col in numeric_cols
        if df_transformed[col].nunique() > 2
    ]

    skew_before = (
        df_transformed[continuous_cols]
        .skew()
    )

    transformed_cols = []

    for col in continuous_cols:

        skew_value = skew_before[col]

        if abs(skew_value) > skew_threshold:

            # Shift values when zero/negative values exist
            shift = 0

            if (df_transformed[col] <= 0).any():

                shift = (
                    abs(df_transformed[col].min()) + 1
                )

                df_transformed[col] = (
                    df_transformed[col] + shift
                )

            # Strong right skew
            if skew_value > 1:

                df_transformed[col] = np.log1p(
                    df_transformed[col]
                )

                method = "log1p"

            # Strong left skew
            elif skew_value < -1:

                df_transformed[col] = np.power(
                    df_transformed[col],
                    2
                )

                method = "square"

            # Moderate skew
            else:

                df_transformed[col] = np.sqrt(
                    df_transformed[col]
                )

                method = "sqrt"

            transformed_cols.append(
                (col, method)
            )

    # Calculate skewness after transformation
    skew_after = (
        df_transformed[continuous_cols]
        .skew()
    )

    summary = pd.DataFrame({
        "Before_Skew": skew_before,
        "After_Skew": skew_after,
        "Change": skew_before - skew_after
    }).round(3)

    logger.info(
        "Skewness correction applied to: %s",
        [c for c, _ in transformed_cols]
    )

    return df_transformed, summary


def run():

    logger.info(
        "Starting outlier detection and skewness correction"
    )

    # ---------------------------------------------------------
    # 1. Read data produced by BasicStats
    # ---------------------------------------------------------

    source = os.path.join(
        OUTPUT_DIR,
        "ingested_cleaned.pkl"
    )

    if not os.path.exists(source):
        raise FileNotFoundError(
            "ingested_cleaned.pkl not found. "
            "Run BasicStats first."
        )

    df = pd.read_pickle(source)

    rows_before = len(df)

    # ---------------------------------------------------------
    # 2. Remove outliers
    # ---------------------------------------------------------

    df_clean, outlier_summary = handle_outliers_remove(
        df,
        skew_threshold=1.0,
        winsor_limits=(0.01, 0.01),
        fill_with="median"
    )

    rows_after_outlier = len(df_clean)

    # ---------------------------------------------------------
    # 3. Correct skewness
    # ---------------------------------------------------------

    df_transformed, skew_summary = auto_fix_skewness(
        df_clean,
        skew_threshold=0.5
    )

    # ---------------------------------------------------------
    # 4. Save processed dataset
    # ---------------------------------------------------------

    output_data = os.path.join(
        OUTPUT_DIR,
        "outlier_skewness_data.pkl"
    )

    df_transformed.to_pickle(output_data)

    # ---------------------------------------------------------
    # 5. Save outlier audit
    # ---------------------------------------------------------

    outlier_summary.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "outlier_summary.csv"
        ),
        index=False
    )

    # ---------------------------------------------------------
    # 6. Save skewness audit
    # ---------------------------------------------------------

    skew_summary.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "skewness_summary.csv"
        )
    )

    # ---------------------------------------------------------
    # 7. Save row-count audit
    # ---------------------------------------------------------

    audit = pd.DataFrame([{
        "rows_before": rows_before,
        "rows_after_outlier_removal": rows_after_outlier,
        "rows_removed": rows_before - rows_after_outlier,
        "percentage_removed": round(
            100 * (rows_before - rows_after_outlier)
            / rows_before,
            2
        )
    }])

    audit.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "outlier_row_audit.csv"
        ),
        index=False
    )

    logger.info(
        "Outlier/skewness processing completed: "
        "%d -> %d rows",
        rows_before,
        rows_after_outlier
    )

    return df_transformed
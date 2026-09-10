"""Main Prefect flow for the Bank Marketing DataOps pipeline.

Runs the data ingestion, preprocessing, EDA and ML feature-importance steps
sequentially and can be served/deployed on a 2-minute schedule.
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tasks"))

from prefect import flow, task, get_run_logger
import BasicStats
import Encoding
import Normalization
import Binning
import CorrelationCoefficient
import PearsonCorrelation
import Visualization
import FeatureImportanceMLAlgorithms
import OutlierSkewness

@task(name="data-ingestion-and-basic-stats", retries=1, retry_delay_seconds=5)
def run_basic_stats():
    logger = get_run_logger()
    logger.info("Starting data ingestion and basic statistics")
    result = BasicStats.run()
    logger.info("Completed ingestion: %s", result)
    return result

@task(name="outlier-skewness",retries=1, retry_delay_seconds=5)
def run_outlier_skewness():
    logger = get_run_logger()
    logger.info("Remove Outliers if any and handle Skewness")
    result = OutlierSkewness.run()
    logger.info("Completed Outlier_Skewness task: %s", result)
    return result

@task(name="categorical-encoding")
def run_encoding():
    logger = get_run_logger()
    result = Encoding.run()
    logger.info("Encoding completed: shape=%s", result.shape)
    return result.shape


@task(name="normalization")
def run_normalization():
    logger = get_run_logger()
    result = Normalization.run()
    logger.info("Normalization completed: shape=%s", result.shape)
    return result.shape


@task(name="binning")
def run_binning():
    logger = get_run_logger()
    result = Binning.run()
    logger.info("Binning completed: shape=%s", result.shape)
    return result.shape


@task(name="correlation-analysis")
def run_correlation():
    logger = get_run_logger()
    result = CorrelationCoefficient.run()
    logger.info("Correlation analysis completed: matrix shape=%s", result.shape)
    return result.shape


@task(name="pearson-correlation")
def run_pearson():
    logger = get_run_logger()
    result = PearsonCorrelation.run()
    logger.info("Selected Pearson correlations completed: %s", result)
    return result


@task(name="visualization")
def run_visualization():
    logger = get_run_logger()
    result = Visualization.run()
    logger.info("Visualizations completed: %s", result)
    return result


@task(name="gradient-boosting-feature-importance")
def run_feature_importance():
    logger = get_run_logger()
    result = FeatureImportanceMLAlgorithms.run()
    logger.info("Gradient Boosting completed: %s", result)
    return result


@flow(name="bank-marketing-dataops-pipeline", log_prints=True)
def main_flow():
    logger = get_run_logger()
    logger.info("Starting Bank Marketing DataOps pipeline")
    run_basic_stats()
    run_outlier_skewness()
    run_encoding()
    run_normalization()
    run_binning()
    run_correlation()
    run_pearson()
    run_visualization()
    run_feature_importance()
    logger.info("Bank Marketing DataOps pipeline completed successfully")


if __name__ == "__main__":
    main_flow.serve(name="bank-marketing-dataops-local", interval=900)
    #main_flow()

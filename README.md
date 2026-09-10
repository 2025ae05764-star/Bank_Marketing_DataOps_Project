# Assignment I — Bank Marketing DataOps Pipeline

This project adapts the webinar `04_dataops_pipeline` structure to the **same dataset and ML work used in the MTech ML Lab notebook**:

- Dataset: `bank-additional-full.xlsx`
- Target: `y` (yes/no)
- Primary ML algorithm for Sub-Objective 1: **Gradient Boosting Classifier**, which was the best-performing model in the supplied notebook.
- Orchestration: **Prefect**
- Schedule: **every 2 minutes (120 seconds)**
- Logging: Python logging + Prefect task/flow logs

> The original dataset file is intentionally not recreated or replaced. Put the exact `bank-additional-full.xlsx` used in the notebook under `data/`.

## Assignment mapping

| Assignment requirement | Project implementation |
|---|---|
| 1.1 Business Understanding | `docs/business_understanding.md` |
| 1.2 Data Ingestion | `tasks/BasicStats.py` reads `data/bank-additional-full.xlsx` |
| 1.3 Data Pre-processing | `tasks/BasicStats.py`, `Encoding.py`, `Normalization.py`, `Binning.py` |
| 1.4 EDA | `CorrelationCoefficient.py`, `PearsonCorrelation.py`, `Visualization.py`, `Binning.py`, `FeatureImportanceMLAlgorithms.py` |
| 1.5 DataOps | `flows/workflow_bankmarketing.py` + Prefect schedule |
| Monitoring/logging | Prefect UI/Cloud + `logging` in every task |
| ML feature importance | `FeatureImportanceMLAlgorithms.py` using Gradient Boosting |
| API access | `flows/flowAPI.py` and `flows/deploymentAPI.py` |

## Project structure

```text
04_dataops_pipeline/
├── flows/
│   ├── workflow_bankmarketing.py
│   ├── deploymentAPI.py
│   ├── flowAPI.py
│   └── config.py
├── tasks/
│   ├── BasicStats.py
│   ├── Encoding.py
│   ├── Normalization.py
│   ├── Binning.py
│   ├── CorrelationCoefficient.py
│   ├── PearsonCorrelation.py
│   ├── Visualization.py
│   └── FeatureImportanceMLAlgorithms.py
├── data/
│   ├── bank-additional-full.xlsx   # place the exact notebook dataset here
│   └── README.md
├── output/
├── docs/
│   └── business_understanding.md
├── prefect.yaml
├── requirements.txt
├── .env.example
├── .gitignore
└── .prefectignore
```

## Setup

```bash
cd 04_dataops_pipeline
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
# source .venv/bin/activate

pip install -r requirements.txt
```

Copy the exact dataset used by the notebook:

```text
data/bank-additional-full.xlsx
```

## Run tasks individually

```bash
python tasks/BasicStats.py
python tasks/Encoding.py
python tasks/Normalization.py
python tasks/Binning.py
python tasks/CorrelationCoefficient.py
python tasks/PearsonCorrelation.py
python tasks/Visualization.py
python tasks/FeatureImportanceMLAlgorithms.py
```

## Run the full flow once

From the project root:

```bash
python -c "from flows.workflow_bankmarketing import main_flow; main_flow()"
```

## Run every 2 minutes locally

```bash
python flows/workflow_bankmarketing.py
```

This uses Prefect `serve()` with a 120-second interval.

## Prefect Cloud deployment

1. Configure `.env` from `.env.example`.
2. Update the repository URL and work-pool name in `prefect.yaml`.
3. Log in to Prefect Cloud.
4. Deploy using the Prefect CLI.

The schedule in `prefect.yaml` is 120 seconds and can be shown in the Prefect dashboard.

## API demonstration for Sub-Objective 2

After deployment, set:

```text
PREFECT_API_URL
PREFECT_API_KEY
PREFECT_ACCOUNT_ID
PREFECT_WORKSPACE_ID
PREFECT_FLOW_ID
PREFECT_DEPLOYMENT_ID
```

Then run:

```bash
python flows/flowAPI.py
python flows/deploymentAPI.py
```

For the report, capture successful responses showing at least four application details such as flow name/id, deployment name/id, schedule, work pool, tags and recent run state.

## Outputs

The pipeline writes generated artifacts to `output/`, including:

- correlation heatmap
- numeric histograms
- boxplots
- target distribution
- bivariate plots
- Gradient Boosting feature importance
- preprocessing summaries

## Important consistency note

The supplied notebook uses five classifier headings but its executed model-evaluation cells contain Logistic Regression, Decision Tree, Random Forest, KNN and Gradient Boosting. The notebook's recorded results identify Gradient Boosting as the best overall model, with 92.5% accuracy, 92.0% weighted F1 and 0.9486 cross-validation ROC-AUC. This project therefore uses Gradient Boosting as the primary ML algorithm in the DataOps pipeline.

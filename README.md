# FraudGuard — Distributed Financial Fraud Detection System

**A big-data fraud detection project built with Apache Spark, PySpark, GraphFrames, and Spark MLlib.**

FraudGuard analyzes financial transactions using distributed data processing, account-network analysis, and machine learning. It compares transaction-based models with graph-enhanced models to investigate how relationships between accounts can help identify suspicious financial activity.

[View Architecture](outputs/figures/architecture.png)

---

## Overview

Traditional fraud detection systems often analyze transactions individually. However, suspicious financial activity can also involve networks of connected accounts, money transfers, and coordinated transaction patterns.

FraudGuard addresses this by combining two perspectives:

- **Transaction-level analysis:** Identifies suspicious patterns using transaction amounts, account balances, transaction types, and historical activity.
- **Graph-based analysis:** Represents accounts as vertices and transactions as edges to identify suspicious account relationships and network structures.

The project uses Apache Spark to process transaction data and Spark MLlib to train and evaluate classification models.

## Key Features

- Distributed transaction processing using PySpark.
- Data cleaning, exploratory analysis, and feature engineering.
- Account-network construction and graph analytics using GraphFrames.
- Graph-based feature extraction and integration with transaction features.
- Fraud classification using Spark MLlib.
- Comparison of transaction-only, graph-only, and combined feature models.
- Imbalanced-class handling using class weights and undersampling.
- Model evaluation using fraud-focused metrics.
- Scalability experiments across different dataset sizes.
- Interactive Streamlit dashboard for exploring results.

## System Architecture

```text
             Financial Transaction Data
                         |
                         v
                Spark Data Ingestion
                         |
                         v
                Data Cleaning & EDA
                         |
                         v
                 Feature Engineering
                         |
             +-----------+-----------+
             |                       |
             v                       v
      Transaction Features     Account Graph
                                     |
                                     v
                               GraphFrames
                                     |
                                     v
                              Graph Features
             |                       |
             +-----------+-----------+
                         |
                         v
                 Feature Integration
                         |
                         v
                  Spark MLlib Models
                         |
                         v
                 Model Evaluation
                         |
             +-----------+-----------+
             |                       |
             v                       v
        Result Tables          Streamlit Dashboard
```

### How Spark works in this project

Apache Spark distributes data across partitions and processes those partitions using executors. The driver coordinates execution and creates a Directed Acyclic Graph (DAG) of operations.

Spark transformations are lazy, meaning computation is delayed until an action such as `count()` or `write()` triggers execution. This allows Spark to optimize the execution plan before processing the data.

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Apache Spark 3.5 | Distributed data processing |
| PySpark | Spark integration with Python |
| GraphFrames | Account-network analysis |
| Spark MLlib | Machine learning |
| Pandas & NumPy | Supporting data operations |
| Matplotlib | Data visualization |
| Streamlit | Interactive dashboard |
| Pytest | Automated testing |

**GraphFrames vs GraphX:** This implementation uses GraphFrames, a DataFrame-based graph processing library accessible through PySpark. GraphX is Spark's graph processing API for Scala/Java. This project performs graph analytics through GraphFrames; it is not a native GraphX implementation.

## Graph Analytics

The account network is constructed using:

- **Vertices:** Financial accounts.
- **Edges:** Transactions between accounts.

The project investigates network patterns using:

- In-degree and out-degree.
- PageRank.
- Connected components.
- Community structures.
- Suspicious account relationships.
- High fan-out and high in-degree accounts.
- Transfer-to-cash-out transaction motifs.

These features help identify patterns that may not be visible when analyzing individual transactions alone.

## Machine Learning

FraudGuard compares different feature configurations:

| Model configuration | Description |
|---|---|
| Legitimate-only baseline | Predicts all transactions as legitimate |
| Transaction-only | Uses transaction-level features |
| Graph-only | Uses graph-derived features |
| Combined | Uses transaction and graph features |

The evaluation focuses on:

- Precision
- Recall
- F1-score
- PR-AUC
- ROC-AUC
- False-positive rate
- False-negative count

Accuracy is not treated as the primary metric because fraud is a minority class in the dataset.

### Evaluation safeguards

- A deterministic, stratified 80/20 train-test split is used.
- Sender-history features for held-out transactions are derived from training data only.
- Held-out graph features are generated from the training account graph.
- `isFlaggedFraud` is excluded to avoid using a pre-existing rule as a predictive feature.
- Class weighting and an undersampling helper are available for imbalance handling.

## Dataset

The project supports synthetic PaySim-style transaction data and the original PaySim CSV format.

The included fallback dataset contains:

| Property | Value |
|---|---:|
| Total transactions | 100,000 |
| Legitimate transactions | 99,500 |
| Fraudulent transactions | 500 |
| Fraud proportion | 0.5% |

The data generator can create larger synthetic datasets for experimentation.

**Important:** Synthetic data is useful for development and controlled experiments, but it does not fully represent real-world fraud behavior.

## Interactive Dashboard

FraudGuard includes a Streamlit dashboard that reads previously generated artifacts. It does not launch Spark jobs or retrain models when opened.

Dashboard sections:

1. Executive Overview
2. Transaction Explorer
3. Graph Explorer
4. Model Evaluation
5. Scalability & Performance
6. Pipeline Status & Data Quality

### Run the dashboard

```powershell
.venv\Scripts\python.exe -m streamlit run dashboard\app.py
```

Open `http://localhost:8501` in your browser.

## Project Setup

### Prerequisites

- Python 3.10–3.12 recommended.
- Java compatible with your Spark installation.
- Apache Spark 3.5.
- Git.

### 1. Clone the repository

```bash
git clone https://github.com/prashantyadavv/FraudGuard-Distributed-Fraud-Detection.git
cd FraudGuard-Distributed-Fraud-Detection
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Windows:

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Generate synthetic transaction data

```bash
python data_generator.py --rows 100000
```

### 5. Execute the pipeline

```bash
python src/main.py --skip-scalability
```

The pipeline performs ingestion, preprocessing, feature engineering, graph analysis, model training, evaluation, and output generation.

### Use an original PaySim dataset

Configure your Kaggle credentials and run:

```bash
python download_data.py
```

Then specify the downloaded CSV:

```bash
python src/main.py --data-path data/PS_20174392719_1491204439457_log.csv
```

## Windows Hadoop Configuration

Some Windows Spark installations require Hadoop filesystem helper binaries.

If Spark reports that `C:\hadoop\bin` is missing, configure compatible `winutils.exe` and `hadoop.dll` files from a trusted university or IT-approved source.

In PowerShell:

```powershell
$env:HADOOP_HOME = "C:\hadoop"
$env:PATH = "C:\hadoop\bin;$env:PATH"
```

Then run:

```powershell
.\.venv\Scripts\python.exe -u src\main.py --skip-scalability
```

## Scalability Experiments

The independent scalability runner measures local Spark aggregation performance across different input sizes.

```powershell
.\.venv\Scripts\python.exe -m src.scalability_experiments --data-path data\synthetic_paysim_100000.csv --sizes 10000 25000 50000 100000
```

Generated files include:

- `scalability_by_size.csv`
- `runtime_by_size.png`
- `throughput_by_size.png`

Results depend on hardware, Spark configuration, execution mode, and warm-up effects. These measurements represent local execution and should not be presented as multi-node cluster benchmarks.

## Output Structure

```text
outputs/
├── dashboard/
├── figures/
├── models/
└── tables/
```

Key outputs include model comparison tables, graph analysis results, suspicious-account reports, runtime measurements, and dashboard artifacts.

## Run Tests

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
```

## Limitations

- The fallback dataset is synthetic and may contain simpler fraud patterns than real financial systems.
- GraphFrames is used instead of native GraphX.
- Local scalability measurements do not demonstrate performance across a multi-node cluster.
- High model scores on synthetic data do not establish real-world fraud detection accuracy.
- Production deployment would require additional testing, real transaction data, monitoring, and security controls.

## Future Improvements

- Implement a native Scala-based GraphX module.
- Evaluate against larger real-world transaction datasets.
- Perform multi-node Spark cluster experiments.
- Explore advanced graph-based machine learning models.
- Add streaming fraud detection using Spark Structured Streaming.
- Introduce model monitoring and concept-drift detection.

## Viva Preparation

Topics demonstrated in this project include:

1. Why Apache Spark is used for distributed processing.
2. Driver, executor, partitions, and DAG execution.
3. Lazy evaluation and transformations versus actions.
4. GraphFrames versus GraphX.
5. PageRank and connected components.
6. Fraud detection using graph relationships.
7. Class imbalance and class weighting.
8. Precision, recall, F1-score, PR-AUC, and ROC-AUC.
9. Data leakage and train-test separation.
10. Scalability benchmarking.

---

**Project:** FraudGuard — Distributed Financial Fraud Detection System  
**Repository:** [GitHub](https://github.com/prashantyadavv/FraudGuard-Distributed-Fraud-Detection)

*Built as a Cluster Computing project to explore distributed financial data processing, graph analytics, and machine learning.*

# Distributed Financial Fraud Detection System

A complete Cluster Computing project using PySpark 3.5, GraphFrames, and Spark MLlib. It analyzes PaySim transactions as both rows and an account network, then compares classifiers with and without graph features.

![Architecture](outputs/figures/architecture.png)

## Architecture

`Transaction data → Spark ingestion → cleaning/features → GraphFrames account graph → graph features merged back → MLlib classifier → evaluation/outputs`

Spark has one **driver** (plans work) and one or more **executors** (process partitions). A DataFrame is a distributed table split into partitions. Transformations are lazy: Spark records them as a DAG, then optimizes and executes it only at an action such as `count()` or `write()`. This matters for transaction-scale data because CPU and memory work can be parallelized instead of loading the whole CSV in one Python process.

GraphX is available only to Scala/Java Spark programs. This PySpark project uses **GraphFrames**, the practical GraphX-equivalent API, while still using Spark DataFrames underneath.

## Setup

Local (Python 3.10--3.12 is recommended; PySpark currently may not support Python 3.13):

```bash
cd fraud-detection-spark
python -m venv .venv
# Windows: .venv\Scripts\activate
pip install -r requirements.txt
python data_generator.py --rows 100000
python src/main.py --skip-scalability
```

For PaySim, install/configure the Kaggle CLI token then run `python download_data.py`; pass its CSV with `--data-path`.

Google Colab:

```python
!pip install pyspark==3.5.3 graphframes==0.6 pandas matplotlib kaggle
!git clone <your-repository-url>
%cd fraud-detection-spark
!python data_generator.py --rows 100000
!python src/main.py --skip-scalability
```

The PyPI `graphframes==0.6` dependency is only the Python wrapper. The Spark session explicitly requests the actual graph engine, `graphframes:graphframes:0.8.3-spark3.5-s_2.12`, and sets a checkpoint directory for Connected Components.

### Windows-only prerequisite: `winutils.exe`

Some Windows Hadoop/Spark installations need the filesystem helpers named
`winutils.exe` and `hadoop.dll`. If startup says that `C:\hadoop\bin` does not
exist, Spark has not processed any transaction data yet. Create `C:\hadoop\bin`,
then place Hadoop 3.3.x-compatible copies of both files there from a source
approved by your university or IT team. Set these PowerShell-session variables
and rerun:

```powershell
$env:HADOOP_HOME = "C:\hadoop"
$env:PATH = "C:\hadoop\bin;$env:PATH"
.\.venv\Scripts\python.exe -u src\main.py --skip-scalability
```

## Run options

```bash
python src/main.py --data-path data/PS_20174392719_1491204439457_log.csv
python src/main.py --sample-fraction 0.1 --skip-scalability
python data_generator.py --rows 500000
```

Outputs are written under `outputs/tables`, `outputs/figures`, and `outputs/models`. The scalability runner records stage runtime; repeat commands for 100K, 500K, 1M, and 3M while changing `local[*]` / shuffle partitions in `create_spark` to populate the complete experiment matrix.

## FraudGuard dashboard

The Streamlit dashboard reads saved artifacts only: it never launches Spark,
recomputes graph analysis, or retrains models. A completed pipeline writes a
manifest, bounded graph data, held-out predictions, model metrics, and timing
data under `outputs/dashboard/`.

```powershell
$env:HADOOP_HOME = "C:\hadoop"
$env:PATH = "C:\hadoop\bin;$env:PATH"
.\.venv\Scripts\python.exe -u src\main.py --skip-scalability
.\.venv\Scripts\python.exe -m streamlit run dashboard\app.py
```

Dashboard pages are Executive Overview, Transaction Explorer, Graph Explorer,
Model Evaluation, Scalability & Performance, and Pipeline Status & Data Quality.
When a required artifact is absent, the page shows instructions rather than
inventing values or claims.

## Evaluation reliability

The pipeline performs a deterministic stratified 80/20 evaluation split before
training graph features. Sender history for held-out rows comes only from the
training partition; graph features for held-out rows come only from the
training account graph. This prevents test labels, future test activity, and
test graph relationships from entering training features. The generated data's
fraud constructions can still be easier than real fraud; perfect scores must be
treated as a synthetic-data limitation, not as proof of production readiness.

The comparison includes a legitimate-only baseline and transaction-only,
graph-only, and combined transaction-plus-graph feature variants. The primary
metrics are fraud-class precision, recall, F1, PR-AUC, ROC-AUC, false-positive
rate, and false-negative count; accuracy is deliberately not used as the lead
metric.

## Run a real scalability benchmark

The normal pipeline saves its own stage timings. Run the independent benchmark
only when you choose to measure it; it records actual local Spark aggregation
timings and does not claim multi-node performance.

```powershell
.\.venv\Scripts\python.exe -m src.scalability_experiments `
  --data-path data\synthetic_paysim_100000.csv `
  --sizes 10000 25000 50000 100000
```

It writes `scalability_by_size.csv`, `runtime_by_size.png`, and
`throughput_by_size.png`. Hardware, master mode, and shuffle partition settings
are recorded with the measurements.

## GraphFrames and GraphX

This project uses **GraphFrames**, not GraphX. GraphX is Spark's Scala/Java API;
GraphFrames is a separate DataFrame-based graph library that can be used from
PySpark. It satisfies the graph-analysis learning objective (degrees, PageRank,
components, communities, and motifs), but it is not a GraphX implementation.
A genuine GraphX deliverable would require adding a separate Scala/Spark module;
that should be approved by an instructor rather than replacing this working
Windows/Python pipeline.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
```

## Sample fallback-data result

The included `synthetic_paysim_100000.csv` contains **100,000** transactions: **99,500 legitimate** and **500 fraud** (0.5%). After a Spark run, `model_comparison_with_graph.csv`, `model_comparison_without_graph.csv`, `suspicious_mules/`, `top_accounts_degree/`, and `runtime_stages.csv` provide the requested final summary from your machine. Scores intentionally are not pre-filled: they depend on your Spark/GraphFrames version and train/test time split.

## Methods and safeguards

- Cleaning removes nulls/duplicates and negative amounts. EDA gives transaction/fraud rates by type, time trends, and high-volume accounts.
- Features include balance errors, drained accounts, amount/balance ratio, time, merchant flag, and Window-based sender history.
- Graph analysis finds degree, PageRank, components, communities, rings, high-in-degree mule accounts, fan-out, and TRANSFER→CASH_OUT motifs.
- `isFlaggedFraud` is excluded because it is a pre-existing rule and would leak a label-like signal. The project uses a deterministic stratified split so the rare fraud class is represented in both train and test; this avoids invalid zero-positive test metrics in the generated dataset.
- Severe imbalance is handled with `weightCol`; an undersampling helper is included for comparison. AUC-PR, precision and recall are central because high accuracy can simply mean predicting every transaction as legitimate.

## How to explain this project in a viva

1. **Why Spark?** It distributes large transaction processing across partitions and avoids single-machine memory limits.
2. **GraphX vs GraphFrames?** GraphX is Scala/Java; GraphFrames exposes analogous graph algorithms to PySpark.
3. **What is PageRank?** It scores accounts important because important accounts connect to them.
4. **Connected components?** Groups of accounts linked by any transaction path.
5. **Why class weights?** Fraud is rare, so positive mistakes need more training influence.
6. **Why is accuracy misleading?** A 0.5% fraud dataset can get 99.5% accuracy by detecting none.
7. **Precision vs recall?** Precision limits false alerts; recall catches more fraud.
8. **AUC-PR vs AUC-ROC?** PR is more revealing when positives are rare.
9. **What is leakage?** Using information unavailable at prediction time, making results unrealistically good.
10. **What is a partition?** One independently processed slice of distributed data.
11. **What is lazy evaluation?** Spark delays transformations until an action, enabling DAG optimization.
12. **Why graph features?** Fraud rings, mules, and hubs are patterns across accounts, not a single row.

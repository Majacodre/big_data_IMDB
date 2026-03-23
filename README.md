# External Data Enrichment for Robust Movie Success Prediction on Noisy IMDB Data

This repository contains our Big Data course project work, where we study whether we can improve movie success prediction from noisy IMDB metadata by adding external context from Rotten Tomatoes.

In short: yes, external enrichment helped us a lot.

---

## 1) Project motivation

The original IMDB dataset is fragmented across multiple files and includes several data quality issues (`\N` sentinels, missing fields, encoding noise, small schema inconsistencies). The target is a binary label (`True`/`False`) indicating movie success.

Our main question was:

**Can external critic/audience signals (Rotten Tomatoes) improve predictive performance compared to using IMDB-only metadata?**

---

## 2) What this repo includes

### Core scripts

- `main.py`: end-to-end pipeline with a scikit-learn/XGBoost RT model.
- `main_pyspark.py`: end-to-end pipeline using PySpark GBT for both baseline and RT-enriched models.

### Utility modules (`utils/`)

- `fetch_files.py` – downloads raw IMDB project files from the course source.
- `merge_files.py` – merges the train shards and joins writing/directing JSON sources.
- `cleaning.py` – DuckDB profiling + PySpark cleaning/imputation.
- `features.py` – feature engineering (including out-of-fold Bayesian encoding for directors/writers).
- `merge_rt.py` / `merge_rt_pyspark.py` – Rotten Tomatoes enrichment (exact + fuzzy matching).
- `model_baseline*.py` – IMDB-only baseline models.
- `model_rt*.py` – RT-enriched models.

### Data and outputs

- `data/` contains intermediate and final datasets.
- `submissions/` contains prediction files formatted for evaluation.
- `pre_cleaning_vis.ipynb` contains exploratory/diagnostic visual work before final cleaning decisions.

---

## 3) Pipeline overview

The workflow is:

1. **(Optional) Download raw files**
2. **Merge IMDB sources** (train shards + writing/directing metadata)
3. **Profile quality with DuckDB SQL**
4. **Clean/impute with PySpark** (medians computed on train split only)
5. **Engineer features**
	- `log_numvotes`
	- director/writer success rates with smoothing
	- out-of-fold encoding for train to reduce leakage
6. **Enrich with Rotten Tomatoes**
	- Stage 1: exact match on normalized title + year (DuckDB)
	- Stage 2: fuzzy fallback via Levenshtein distance + year window (PySpark)
	- genre one-hot encoding and RT score imputation from train medians
7. **Train models + export predictions**

---

## 4) Environment setup

This project is Python-based and uses a local virtual environment.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install pandas pyspark duckdb scikit-learn xgboost requests
```

On Windows (PowerShell), activate with:

```powershell
.venv\Scripts\Activate.ps1
```

If PySpark has Java-related startup issues, make sure a JDK is installed and available in your shell.

---

## 5) Run instructions

### Option A — PySpark workflow (recommended)

```bash
python main_pyspark.py
```

This runs:

- profiling, cleaning, feature engineering
- Rotten Tomatoes merge
- baseline + RT-enriched PySpark model training
- submission export for validation/test

### Option B — alternative workflow (scikit-learn/XGBoost RT model)

```bash
python main.py
```

This follows the same preprocessing/enrichment pipeline but trains the RT model with XGBoost.

---

## 6) Main generated files

Typical outputs include:

- `data/clean_train.csv`, `data/clean_validation.csv`, `data/clean_test.csv`
- `data/features_train.csv`, `data/features_validation.csv`, `data/features_test.csv`
- `data/rt_train.csv`, `data/rt_validation.csv`, `data/rt_test.csv`
- `data/baseline_feature_importance_results.csv`
- `data/rt_feature_importance_results.csv`
- prediction files in `submissions/`

---

## 7) Notes on modeling choices

- We intentionally compute imputation statistics from **train only** and apply them to validation/test.
- Director/writer rates use **Bayesian smoothing** to stabilize rare names.
- For train encoding, we use an **OOF-style adjustment** to reduce leakage from row self-counting.
- RT join is two-stage to balance precision and recall under noisy titles.

---

## 8) Practical caveats

- Some scripts still contain commented alternatives from experimentation; this is expected for a research project repository.
- The `data/` folder is currently versioned with many intermediate artifacts for reproducibility during grading/reporting.
- Runtime can vary significantly depending on Spark memory settings and local hardware.

---

## 9) Project context

This codebase was built for a Big Data group project on noisy real-world integration.

Concretely, the technical focus was:

- distributed preprocessing and cleaning with PySpark
- leakage-aware feature engineering (train-only stats + OOF-style encoding)
- two-stage external data matching (exact + fuzzy)
- comparing IMDB-only vs RT-enriched performance

If you are reviewing this repo for the first time, start with `main_pyspark.py`, then inspect `utils/features.py` and `utils/merge_rt.py` for the key contributions.
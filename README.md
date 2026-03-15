# big_data_IMDB

## Environment (latest stable)

This project is now configured for:

- Python `3.13.12`
- PySpark `4.1.1`
- DuckDB `1.5.0`
- Pandas `3.0.1`

### Option A — Conda (recommended)

```bash
conda env create -f environment.yml
conda activate bigdata-imdb-py313
```

Run with consistent Spark Python versions:

```bash
PYSPARK_PYTHON=$(which python) \
PYSPARK_DRIVER_PYTHON=$(which python) \
python main.py
```

### Option B — Pip only

```bash
python -m pip install -r requirements.txt
```

## Notes

- If you have multiple Anaconda installs, always run with the same interpreter for driver and workers (set both `PYSPARK_PYTHON` and `PYSPARK_DRIVER_PYTHON`).
- Current runtime issue after env upgrade is data-related (`label` contains nulls), not package-version related.
import duckdb
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, FloatType
import unicodedata
import re


# PROFILING: on duckDB only as it's ideal for fast analysis over a single table

def profile_data(csv_path: str):
    """
    Prints a data quality report, using only duckDB with SQL queries for the initial merged dataset
    """
    con = duckdb.connect()
    con.execute(f"CREATE TABLE data AS SELECT * FROM read_csv_auto('{csv_path}')")

    # full row count
    print("ROW COUNT")
    con.execute("SELECT COUNT(*) AS total_rows FROM data").df().pipe(print)  # way to easily print queries result
    print("")


    # real nulls per column
    print("NULL COUNTS PER COLUMN")
    null_query = " UNION ALL ".join([
        f"SELECT '{col}' AS column, COUNT(*) - COUNT({col}) AS nulls "
        f"FROM data"
        for col in con.execute("DESCRIBE data").df()["column_name"].tolist()
    ])
    con.execute(null_query).df().pipe(print)
    print("")

    # fake-missing \N counts in string columns (sentinels)
    print(r"'\N' SENTINEL COUNTS (in selected columns)")
    for col in ["startYear", "endYear", "runtimeMinutes", "writers", "directors"]:
        count = con.execute(
            f"SELECT COUNT(*) FROM data WHERE {col} = '\\N'"
        ).fetchone()[0]
        print(f"  {col}: {count} \\N values")
    print("")

    # duplicate IDs (should be zero)
    print("DUPLICATE tconst")
    dups = con.execute(
        "SELECT COUNT(*) FROM (SELECT tconst FROM data GROUP BY tconst HAVING COUNT(*) > 1)"
    ).fetchone()[0]
    print(f"  Duplicate tconst: {dups}")
    print("")

    # numeric outliers after eliminating sentinels
    print("startYear ANOMALIES (after removing \\N)")
    con.execute("""
        CREATE TABLE start_clean AS
        SELECT TRY_CAST(NULLIF(startYear, '\\N') AS INTEGER) AS yr FROM data
    """)
    con.execute("""
        SELECT
            SUM(CASE WHEN yr < 1880 THEN 1 ELSE 0 END) AS before_1880,
            SUM(CASE WHEN yr > 2025 THEN 1 ELSE 0 END) AS after_2025,
            MIN(yr) AS min_year,
            MAX(yr) AS max_year
        FROM start_clean
    """).df().pipe(print)
    print("")

    print("runtimeMinutes ANOMALIES (after removing \\N)")
    con.execute("""
        CREATE TABLE runtime_clean AS
        SELECT TRY_CAST(NULLIF(runtimeMinutes, '\\N') AS FLOAT) AS rt FROM data
    """)
    con.execute("""
        SELECT
            SUM(CASE WHEN rt <= 0  THEN 1 ELSE 0 END) AS lte_zero,
            SUM(CASE WHEN rt > 600 THEN 1 ELSE 0 END) AS gt_600,
            MIN(rt) AS min_runtime,
            MAX(rt) AS max_runtime
        FROM runtime_clean
    """).df().pipe(print)
    print("")

    print("numVotes ANOMALIES")
    con.execute("""
        SELECT
            COUNT(*) FILTER (WHERE numVotes IS NULL)  AS null_votes,
            COUNT(*) FILTER (WHERE numVotes < 0)      AS negative_votes,
            MIN(numVotes) AS min_votes,
            MAX(numVotes) AS max_votes
        FROM data
    """).df().pipe(print)
    print("")

    # target balance
    print("LABEL DISTRIBUTION")
    con.execute(
        "SELECT label, COUNT(*) AS count FROM data GROUP BY label ORDER BY label"
    ).df().pipe(print)
    print("")

    # string errors
    print("NON-ASCII TITLES (accent errors)")
    count = con.execute(
        r"SELECT COUNT(*) FROM data WHERE regexp_matches(primaryTitle, '[^\x00-\x7F]')"
    ).fetchone()[0]
    print(f"  primaryTitle with non-ASCII chars: {count}")
    print("  Sample:")
    con.execute(
        r"SELECT tconst, primaryTitle FROM data WHERE regexp_matches(primaryTitle, '[^\x00-\x7F]') LIMIT 5"
    ).df().pipe(print)
    print("")

    # nulls on original title
    print("originalTitle NULLs")
    con.execute(
        "SELECT COUNT(*) AS null_originalTitle FROM data WHERE originalTitle IS NULL"
    ).df().pipe(print)
    print("")

    con.close()


# CLEANING: on PySpark for distributed imputation across splits
# use of Spark DataFrame API (instead of RDDs) for typing columns, SQL transformations and working on structured data...

def clean_data(train_csv: str, val_csv: str, test_csv: str, train_out: str, val_out: str, test_out: str):
    """
    Cleans and imputes the original merged CSVs. Note that medians are computed from the train split only and applied to all three.
    """

    spark = SparkSession.builder \
        .master("local") \
        .config("spark.driver.bindAddress", "127.0.0.1") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    # load splits
    train = spark.read.csv(train_csv, header=True, inferSchema=True)
    val = spark.read.csv(val_csv, header=True, inferSchema=True)
    test = spark.read.csv(test_csv, header=True, inferSchema=True)

    def apply_cleaning(df, medians: dict):
        """Applies all cleaning steps to a split"""

        # drop index column if present
        if "column0" in df.columns:
            df = df.drop("column0")

        # replace \N sentinel by real nulls
        for col in ["startYear", "endYear", "runtimeMinutes", "writers", "directors"]:
            if col in df.columns:
                df = df.withColumn(col, F.when(F.col(col) == r"\N", None).otherwise(F.col(col)))

        # correct data types
        df = df.withColumn("startYear", F.col("startYear").cast(IntegerType()))
        df = df.withColumn("runtimeMinutes", F.col("runtimeMinutes").cast(FloatType()))
        df = df.withColumn("numVotes", F.col("numVotes").cast(FloatType()))

        # restrict year to [1880, 2025] (may not be needed according to profiling, but good sanity check for test or val)
        df = df.withColumn(
            "startYear",
            F.when(F.col("startYear") < 1880, None)
            .when(F.col("startYear") > 2025, None)
            .otherwise(F.col("startYear"))
        )

        # restrict runtime to (0, 600] minutes (again, should not be needed but good sanity check)
        df = df.withColumn(
            "runtimeMinutes",
            F.when(F.col("runtimeMinutes") <= 0,   None)
            .when(F.col("runtimeMinutes") > 600,   None)
            .otherwise(F.col("runtimeMinutes"))
        )

        # treat negative numVotes as null (shouldn't be any according to profiling, but just in case)
        df = df.withColumn(
            "numVotes",
            F.when(F.col("numVotes") < 0, None).otherwise(F.col("numVotes"))
        )

        # impute numeric columns with train medians
        df = df.withColumn(
            "startYear",
            F.coalesce(F.col("startYear"), F.lit(medians["startYear"]))
        )
        df = df.withColumn(
            "runtimeMinutes",
            F.coalesce(F.col("runtimeMinutes"), F.lit(medians["runtimeMinutes"]))
        )
        df = df.withColumn(
            "numVotes",
            F.coalesce(F.col("numVotes"), F.lit(medians["numVotes"]))
        )

        # merge startYear and endYear as they just refer to year (induced error)
        df = df.withColumn("startYear", F.coalesce(F.col("startYear"), F.col("endYear")))
        df = df.drop("endYear")
        df = df.withColumnRenamed("startYear", "year")

        # put primaryTitle as originalTitle if originalTitle is null
        df = df.withColumn("originalTitle", F.coalesce(F.col("originalTitle"), F.col("primaryTitle")))

        # if title is the same (maybe useful for the model)
        df = df.withColumn("title_is_same", F.when(F.col("primaryTitle") == F.col("originalTitle"), 1).otherwise(0))

        # normalize title strings
        def normalize_title(title):
            if title is None:
                return None
            # drop accents
            title = unicodedata.normalize("NFD", title)
            title = "".join(c for c in title if unicodedata.category(c) != "Mn")
            # lowercase
            title = title.lower()
            # remove punctuation and extra whitespace
            title = re.sub(r"[^\w\s]", "", title)
            title = re.sub(r"\s+", " ", title).strip()
            return title

        normalize_udf = F.udf(normalize_title)
        df = df.withColumn("normalized_title", normalize_udf(F.col("primaryTitle")))

        # from bool to int for the model
        if "label" in df.columns: # only on train
            df = df.withColumn("label", F.col("label").cast(IntegerType()))

        # drop duplicates (shouldn't be any according to profiling, but just in case)
        df = df.dropDuplicates(["tconst"])

        return df

    # compute medians from train only
    train_for_median = train
    for col in ["startYear", "runtimeMinutes"]:
        train_for_median = train_for_median.withColumn(
            col, F.when(F.col(col) == r"\N", None).otherwise(F.col(col)).cast(FloatType()))

    median_rows = train_for_median.select(
        F.percentile_approx("startYear", 0.5).alias("startYear"),
        F.percentile_approx("runtimeMinutes", 0.5).alias("runtimeMinutes"),
        F.percentile_approx("numVotes", 0.5).alias("numVotes"),
    ).collect()[0]

    medians = {
        "startYear": int(median_rows["startYear"]),
        "runtimeMinutes": float(median_rows["runtimeMinutes"]),
        "numVotes": float(median_rows["numVotes"]),
    }
    print(f"[INFO] Train medians: {medians}")

    # apply cleaning to each split
    train_clean = apply_cleaning(train, medians)
    val_clean = apply_cleaning(val, medians)
    test_clean = apply_cleaning(test, medians)

    # save outputs
    train_clean.toPandas().to_csv(train_out, index=False)
    val_clean.toPandas().to_csv(val_out, index=False)
    test_clean.toPandas().to_csv(test_out, index=False)

    print(f"[INFO] Clean train saved to: {train_out}")
    print(f"[INFO] Clean validation saved to: {val_out}")
    print(f"[INFO] Clean test saved to: {test_out}")

    spark.stop()
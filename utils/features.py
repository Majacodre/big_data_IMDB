import math
import glob
import shutil
import os
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import FloatType


def build_features(
    train_csv: str,
    val_csv:   str,
    test_csv:  str,
    train_out: str,
    val_out:   str,
    test_out:  str,
) -> None:
    """
    Engineers features for the IMDB binary classification task.
    All rates are computed from train only and applied to val/test
    to avoid data leakage.

    Features produced:
        log_numvotes              log(numVotes + 1) — compresses skewed vote scale
        director_success_rate     % of highly rated films per director
        director_movie_count      how prolific the director is
        writer_success_rate       % of highly rated films per writer (avg across writers)
        writer_movie_count        avg movie count across writers
    """

    spark = (
        SparkSession.builder
        .master("local[*]")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.driver.memory", "8g")
        .config("spark.executor.memory", "8g")
        .config("spark.sql.shuffle.partitions", "4")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    train = spark.read.csv(train_csv, header=True, inferSchema=True)
    val   = spark.read.csv(val_csv,   header=True, inferSchema=True)
    test  = spark.read.csv(test_csv,  header=True, inferSchema=True)

    # ---------------------------------------------------------------- #
    # log_numvotes                                                        #
    # numVotes ranges from ~1K to 2.5M — log-transform compresses the   #
    # scale so the model isn't dominated by blockbuster outliers.        #
    # ---------------------------------------------------------------- #
    log_udf = F.udf(
        lambda v: float(math.log1p(v)) if v is not None else None,
        FloatType()
    )

    # ---------------------------------------------------------------- #
    # Director success rate                                              #
    # Each movie has at most one director in this dataset.              #
    # We compute: (# highly rated movies directed) / (# movies directed)#
    # ---------------------------------------------------------------- #
    director_stats = (
        train
        .filter(F.col("directors").isNotNull())
        .groupBy("directors")
        .agg(
            F.count("*").alias("director_movie_count"),
            F.sum(F.col("label").cast("int")).alias("director_hits"),
        )
        .withColumn(
            "director_success_rate",
            F.col("director_hits") / F.col("director_movie_count")
        )
        .select("directors", "director_success_rate", "director_movie_count")
    )

    # Global fallback for unseen directors (train mean)
    director_mean = (
        director_stats
        .select(F.mean("director_success_rate").alias("mean"))
        .collect()[0]["mean"]
    )
    director_count_mean = (
        director_stats
        .select(F.mean("director_movie_count").alias("mean"))
        .collect()[0]["mean"]
    )

    # ---------------------------------------------------------------- #
    # Writer success rate                                                #
    # A movie can have multiple writers (comma-separated).              #
    # Strategy: explode writers → compute per-writer stats →           #
    # average the success rates back per movie.                         #
    # ---------------------------------------------------------------- #
    train_exploded = (
        train
        .filter(F.col("writers").isNotNull())
        .withColumn("writer", F.explode(F.split(F.col("writers"), ",")))
    )

    writer_stats = (
        train_exploded
        .groupBy("writer")
        .agg(
            F.count("*").alias("writer_movie_count"),
            F.sum(F.col("label").cast("int")).alias("writer_hits"),
        )
        .withColumn(
            "writer_success_rate",
            F.col("writer_hits") / F.col("writer_movie_count")
        )
        .select("writer", "writer_success_rate", "writer_movie_count")
    )

    # Global fallback for unseen writers (train mean)
    writer_mean = (
        writer_stats
        .select(F.mean("writer_success_rate").alias("mean"))
        .collect()[0]["mean"]
    )
    writer_count_mean = (
        writer_stats
        .select(F.mean("writer_movie_count").alias("mean"))
        .collect()[0]["mean"]
    )

    print(f"[INFO] Director success rate mean (fallback): {director_mean:.4f}")
    print(f"[INFO] Writer success rate mean (fallback):   {writer_mean:.4f}")

    # ---------------------------------------------------------------- #
    # Apply features to a split                                         #
    # ---------------------------------------------------------------- #
    def apply_features(df):

        # log_numvotes
        df = df.withColumn("log_numvotes", log_udf(F.col("numVotes")))

        # -- Director features --------------------------------------- #
        df = df.join(director_stats, on="directors", how="left")
        df = df.withColumn(
            "director_success_rate",
            F.coalesce(F.col("director_success_rate"), F.lit(director_mean))
        )
        df = df.withColumn(
            "director_movie_count",
            F.coalesce(F.col("director_movie_count"), F.lit(director_count_mean))
        )

        # -- Writer features ----------------------------------------- #
        # Explode writers, join stats, then average back per movie
        df_exploded = (
            df
            .withColumn("writer", F.explode_outer(F.split(F.col("writers"), ",")))
            .join(writer_stats, on="writer", how="left")
            .withColumn(
                "writer_success_rate",
                F.coalesce(F.col("writer_success_rate"), F.lit(writer_mean))
            )
            .withColumn(
                "writer_movie_count",
                F.coalesce(F.col("writer_movie_count"), F.lit(writer_count_mean))
            )
        )

        writer_agg = (
            df_exploded
            .groupBy("tconst")
            .agg(
                F.mean("writer_success_rate").alias("writer_success_rate"),
                F.mean("writer_movie_count").alias("writer_movie_count"),
            )
        )

        df = df.join(writer_agg, on="tconst", how="left")

        return df

    train_feat = apply_features(train)
    val_feat   = apply_features(val)
    test_feat  = apply_features(test)

    # ---------------------------------------------------------------- #
    # Save outputs                                                       #
    # Write directly from Spark workers to disk — avoids pulling all    #
    # data into the driver with toPandas() which caused OOM errors.     #
    # coalesce(1) forces a single output file instead of partitions.    #
    # ---------------------------------------------------------------- #
    def save_single_csv(df, path):
        tmp = path + "_tmp"
        df.coalesce(1).write.csv(tmp, header=True, mode="overwrite")
        part = glob.glob(os.path.join(tmp, "part-*.csv"))[0]
        shutil.move(part, path)
        shutil.rmtree(tmp)

    save_single_csv(train_feat, train_out)
    save_single_csv(val_feat,   val_out)
    save_single_csv(test_feat,  test_out)

    print(f"[INFO] Features train saved to:      {train_out}")
    print(f"[INFO] Features validation saved to: {val_out}")
    print(f"[INFO] Features test saved to:       {test_out}")

    spark.stop()

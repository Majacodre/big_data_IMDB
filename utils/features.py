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

    For TRAIN, we use out-of-fold (OOF) encoding to prevent leakage:
        Each row's success rate excludes that row's own label.
        Formula: (total_hits - row_label + prior * k) / (total_count - 1 + k)
        We store raw hits and counts so the subtraction is exact.

    For VAL/TEST, we use full train stats (no leakage risk).

    Features produced:
        log_numvotes              log(numVotes + 1)
        director_success_rate     Bayesian-smoothed OOF rate for director
        director_movie_count      how prolific the director is
        writer_success_rate       Bayesian-smoothed OOF rate (avg across writers)
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
    # ---------------------------------------------------------------- #
    log_udf = F.udf(
        lambda v: float(math.log1p(v)) if v is not None else None,
        FloatType()
    )

    # ---------------------------------------------------------------- #
    # Global label mean — Bayesian prior                                 #
    # ---------------------------------------------------------------- #
    global_mean = train.select(F.mean(F.col("label").cast("double"))).collect()[0][0]
    SMOOTHING_K = 10
    print(f"[INFO] Global label mean (prior): {global_mean:.4f}")

    # ---------------------------------------------------------------- #
    # Store RAW hits + counts so OOF subtraction is exact               #
    # ---------------------------------------------------------------- #
    director_totals = (
        train
        .filter(F.col("directors").isNotNull())
        .groupBy("directors")
        .agg(
            F.count("*").cast("double").alias("dir_count"),
            F.sum(F.col("label").cast("double")).alias("dir_hits"),
        )
    )

    train_exploded = (
        train
        .filter(F.col("writers").isNotNull())
        .withColumn("writer", F.explode(F.split(F.col("writers"), ",")))
    )

    writer_totals = (
        train_exploded
        .groupBy("writer")
        .agg(
            F.count("*").cast("double").alias("wri_count"),
            F.sum(F.col("label").cast("double")).alias("wri_hits"),
        )
    )

    # Fallback rates for val/test unseen people
    # Use smoothed rate on all of train
    dir_fallback_rate = float(
        director_totals
        .select(
            F.sum("dir_hits") / F.count("dir_hits")
        ).collect()[0][0] or global_mean
    )
    wri_fallback_rate = float(
        writer_totals
        .select(
            F.sum("wri_hits") / F.count("wri_hits")
        ).collect()[0][0] or global_mean
    )
    dir_fallback_count  = float(director_totals.select(F.mean("dir_count")).collect()[0][0])
    wri_fallback_count  = float(writer_totals.select(F.mean("wri_count")).collect()[0][0])

    print(f"[INFO] Director fallback rate: {dir_fallback_rate:.4f}")
    print(f"[INFO] Writer fallback rate:   {wri_fallback_rate:.4f}")

    # ---------------------------------------------------------------- #
    # OOF encoding for TRAIN                                             #
    # rate = (hits - row_label + prior*k) / (count - 1 + k)            #
    # This is exact because we use raw hits/counts, not smoothed rates  #
    # ---------------------------------------------------------------- #
    def apply_oof_features_train(df):
        df = df.withColumn("log_numvotes", log_udf(F.col("numVotes")))

        # -- Director OOF -------------------------------------------- #
        df = df.join(director_totals, on="directors", how="left")
        df = df.withColumn(
            "director_success_rate",
            F.when(
                F.col("dir_count").isNotNull(),
                (F.col("dir_hits") - F.col("label").cast("double") + F.lit(global_mean * SMOOTHING_K))
                / (F.col("dir_count") - F.lit(1.0) + F.lit(SMOOTHING_K))
            ).otherwise(F.lit(dir_fallback_rate))
        )
        df = df.withColumn(
            "director_movie_count",
            F.coalesce(F.col("dir_count") - F.lit(1.0), F.lit(dir_fallback_count))
        )
        df = df.drop("dir_count", "dir_hits")

        # -- Writer OOF ---------------------------------------------- #
        df_exploded = (
            df
            .withColumn("writer", F.explode_outer(F.split(F.col("writers"), ",")))
            .join(writer_totals, on="writer", how="left")
        )
        df_exploded = df_exploded.withColumn(
            "writer_success_rate",
            F.when(
                F.col("wri_count").isNotNull(),
                (F.col("wri_hits") - F.col("label").cast("double") + F.lit(global_mean * SMOOTHING_K))
                / (F.col("wri_count") - F.lit(1.0) + F.lit(SMOOTHING_K))
            ).otherwise(F.lit(wri_fallback_rate))
        )
        df_exploded = df_exploded.withColumn(
            "writer_movie_count",
            F.coalesce(F.col("wri_count") - F.lit(1.0), F.lit(wri_fallback_count))
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

    # ---------------------------------------------------------------- #
    # Standard encoding for VAL/TEST                                     #
    # Use full smoothed train stats — no leakage risk here              #
    # ---------------------------------------------------------------- #
    def apply_inference_features(df):
        df = df.withColumn("log_numvotes", log_udf(F.col("numVotes")))

        # Director
        dir_stats = director_totals.withColumn(
            "director_success_rate",
            (F.col("dir_hits") + F.lit(global_mean * SMOOTHING_K))
            / (F.col("dir_count") + F.lit(SMOOTHING_K))
        ).withColumnRenamed("dir_count", "director_movie_count") \
         .drop("dir_hits")

        df = df.join(dir_stats, on="directors", how="left")
        df = df.withColumn(
            "director_success_rate",
            F.coalesce(F.col("director_success_rate"), F.lit(dir_fallback_rate))
        )
        df = df.withColumn(
            "director_movie_count",
            F.coalesce(F.col("director_movie_count"), F.lit(dir_fallback_count))
        )

        # Writers
        wri_stats = writer_totals.withColumn(
            "writer_success_rate",
            (F.col("wri_hits") + F.lit(global_mean * SMOOTHING_K))
            / (F.col("wri_count") + F.lit(SMOOTHING_K))
        ).withColumnRenamed("wri_count", "writer_movie_count") \
         .drop("wri_hits")

        df_exploded = (
            df
            .withColumn("writer", F.explode_outer(F.split(F.col("writers"), ",")))
            .join(wri_stats, on="writer", how="left")
            .withColumn(
                "writer_success_rate",
                F.coalesce(F.col("writer_success_rate"), F.lit(wri_fallback_rate))
            )
            .withColumn(
                "writer_movie_count",
                F.coalesce(F.col("writer_movie_count"), F.lit(wri_fallback_count))
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

    train_feat = apply_oof_features_train(train)
    val_feat   = apply_inference_features(val)
    test_feat  = apply_inference_features(test)

    # ---------------------------------------------------------------- #
    # Save outputs                                                       #
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
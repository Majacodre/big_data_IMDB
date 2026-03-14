import duckdb
import pandas as pd
import unicodedata
import re
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StringType, IntegerType


# title normalization, same output as in cleaning.py
def normalize_title(title: str) -> str:
    if not isinstance(title, str):
        return None
    # decompose accented chars then strip combining marks
    title = unicodedata.normalize("NFD", title)
    title = "".join(c for c in title if unicodedata.category(c) != "Mn")
    title = title.lower()
    title = re.sub(r"[^\w\s]", "", title)
    title = re.sub(r"\s+", " ", title).strip()
    return title


def merge_with_rotten_tomatoes(
    train_csv:  str,
    val_csv:    str,
    test_csv:   str,
    rt_csv:     str,
    train_out:  str,
    val_out:    str,
    test_out:   str,):
    """
    Merges original IMDB data with Rotten Tomatoes scores and genre
    Two steps strategy:
        Step 1 — DuckDB: exact join on normalized_title + year
                  Justified: fast process SQL for the easy cases (~exact matches)
        Step 2 — PySpark: fuzzy join for unmatched rows using
                  Levenshtein distance <= 1 AND year within +/- 1
                  Justified: distributed cross-join over 143K RT rows
                  is too expensive for a single process — Spark broadcasts
                  the RT table across workers for parallel evaluation.

    RT columns added: tomatoMeter, audienceScore, genre
    """

    # prepare RT data
    rt = pd.read_csv(rt_csv)

    # extract year taking the earliest non-null year from both date columns
    rt["releaseDateTheaters"] = pd.to_datetime(rt["releaseDateTheaters"], errors="coerce")
    rt["releaseDateStreaming"] = pd.to_datetime(rt["releaseDateStreaming"], errors="coerce")
    rt["rt_year"] = rt[["releaseDateTheaters", "releaseDateStreaming"]].min(axis=1).dt.year

    # normalize title
    rt["normalized_title"] = rt["title"].apply(normalize_title)

    # keep only columns needed for enrichment
    rt = rt[["normalized_title", "rt_year", "tomatoMeter", "audienceScore", "genre"]].dropna(
        subset=["normalized_title", "rt_year"]
    )

    # drop duplicate (title, year) pairs — keep row with most scores filled
    rt["score_count"] = rt[["tomatoMeter", "audienceScore"]].notna().sum(axis=1)
    rt = rt.sort_values("score_count", ascending=False).drop_duplicates(
        subset=["normalized_title", "rt_year"]
    ).drop(columns=["score_count"])

    rt["rt_year"] = rt["rt_year"].astype(int)

    ### STEP 1: exact join with DuckDB
    con = duckdb.connect(database=":memory:")

    # register RT table
    con.register("rt_tbl", rt)

    for split_name, csv_path, out_path in [("train", train_csv, train_out), ("val",   val_csv,   val_out), ("test",  test_csv,  test_out),]:
        # Load split and normalize its title
        df = pd.read_csv(csv_path)
        df["normalized_title"] = df["normalized_title"].apply(normalize_title) \
            if "normalized_title" in df.columns \
            else df["primaryTitle"].apply(normalize_title)
        df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")

        con.register("imdb_tbl", df)

        # exact join on normalized_title + year
        matched = con.execute("""
            SELECT i.*, r.tomatoMeter, r.audienceScore, r.genre
            FROM imdb_tbl i
            LEFT JOIN rt_tbl r
              ON i.normalized_title = r.normalized_title
             AND i.year = r.rt_year
        """).df()

        # track which rows are still unmatched (both RT scores null)
        unmatched_mask = matched["tomatoMeter"].isna() & matched["audienceScore"].isna()
        matched_stage1 = matched[~unmatched_mask].copy()
        unmatched_df   = df[unmatched_mask.values].copy()

        print(f"[INFO] {split_name} — Stage 1 exact match: "
              f"{len(matched_stage1)} matched, {len(unmatched_df)} unmatched")

        ### STEP 2: fuzzy join with PySpark
        spark = (
            SparkSession.builder
            .master("local[*]")
            .config("spark.driver.bindAddress", "127.0.0.1")
            .config("spark.driver.memory", "8g")
            .config("spark.sql.shuffle.partitions", "4")
            .getOrCreate()
        )
        spark.sparkContext.setLogLevel("ERROR")

        rt_spark   = spark.createDataFrame(rt)
        unmatched_spark = spark.createDataFrame(unmatched_df)

        # broadcast RT table as it fits in memory on every worker (avoids full shuffle)
        from pyspark.sql.functions import broadcast

        fuzzy = (
            unmatched_spark.alias("i")\
            .join(broadcast(rt_spark.alias("r")), how="inner",
                  on=(
                      # year within ±1
                      (F.abs(F.col("i.year") - F.col("r.rt_year")) <= 1)
                      &
                      # Levenshtein distance <= 1 on normalized titles
                      (F.levenshtein(F.col("i.normalized_title"), F.col("r.normalized_title")) <= 1))))

        # if different RT rows match, keep the closest title (lowest distance)
        fuzzy = fuzzy.withColumn(
            "lev_dist",
            F.levenshtein(F.col("i.normalized_title"), F.col("r.normalized_title"))
        )

        # keep best match per tconst
        from pyspark.sql.window import Window
        w = Window.partitionBy("i.tconst").orderBy("lev_dist")
        fuzzy = (
            fuzzy
            .withColumn("rank", F.row_number().over(w))
            .filter(F.col("rank") == 1)
            .drop("rank", "lev_dist", "r.normalized_title", "r.rt_year")
        )

        # rename RT columns back
        fuzzy = (
            fuzzy
            .withColumnRenamed("r.tomatoMeter",  "tomatoMeter")
            .withColumnRenamed("r.audienceScore", "audienceScore")
            .withColumnRenamed("r.genre",         "genre")
        )

        fuzzy_pd = fuzzy.toPandas()
        spark.stop()

        # drop duplicate columns produced by the Spark join
        fuzzy_pd = fuzzy_pd.loc[:, ~fuzzy_pd.columns.duplicated()]

        # keep only columns from the original split + the 3 RT columns
        rt_cols  = ["tomatoMeter", "audienceScore", "genre"]
        keep     = [c for c in df.columns if c in fuzzy_pd.columns] + \
                [c for c in rt_cols if c in fuzzy_pd.columns]
        fuzzy_pd = fuzzy_pd[keep]

        print(f"[INFO] {split_name} — Stage 2 fuzzy match:  "
              f"{len(fuzzy_pd)} additional matches")

        # COMBINE BOTH STAGES
        # rows that fuzzy matched
        fuzzy_tconsts = set(fuzzy_pd["tconst"].tolist()) if len(fuzzy_pd) > 0 else set()

        # rows still unmatched after both stages — add null RT columns
        still_unmatched = unmatched_df[~unmatched_df["tconst"].isin(fuzzy_tconsts)].copy()
        still_unmatched["tomatoMeter"] = None
        still_unmatched["audienceScore"] = None
        still_unmatched["genre"] = None

        final = pd.concat([matched_stage1, fuzzy_pd, still_unmatched], ignore_index=True)
        # has_rt_match: 1 if the movie was found in RT, 0 if not
        # this can itself a signal: obscure/low-quality movies tend to not be in RT
        final["has_rt_match"] = final["tomatoMeter"].notna().astype(int)

        final.to_csv(out_path, index=False)

        # nulls analysis
        n = len(final)
        print(f"[INFO] {split_name} — Final rows: {n}")
        print(f"[INFO] {split_name} — tomatoMeter  nulls: {final['tomatoMeter'].isna().sum()} "
              f"({final['tomatoMeter'].isna().sum()/n*100:.1f}%)")
        print(f"[INFO] {split_name} — audienceScore nulls: {final['audienceScore'].isna().sum()} "
              f"({final['audienceScore'].isna().sum()/n*100:.1f}%)")
        print(f"[INFO] {split_name} — genre         nulls: {final['genre'].isna().sum()} "
              f"({final['genre'].isna().sum()/n*100:.1f}%)")
        print(f"[INFO] {split_name} — merged CSV saved to: {out_path}")
        print()


    # FINAL CLEANING 
    # load splits back
    train_final = pd.read_csv(train_out)
    val_final = pd.read_csv(val_out)
    test_final = pd.read_csv(test_out)

    # compute medians from train only
    tomato_median = train_final["tomatoMeter"].median()
    audience_median = train_final["audienceScore"].median()
    print(f"[INFO] tomatoMeter median (train):   {tomato_median}")
    print(f"[INFO] audienceScore median (train): {audience_median}")

    # impute all three splits with train medians
    for df in [train_final, val_final, test_final]:
        df["tomatoMeter"] = df["tomatoMeter"].fillna(tomato_median)
        df["audienceScore"] = df["audienceScore"].fillna(audience_median)
        df["genre"] = df["genre"].fillna("Unknown")

    print("[INFO] RT nulls imputed")

    # extract all unique genres from train only — no leakage
    all_genres = set()
    for g in train_final["genre"].dropna():
        for tag in g.split(","):
            all_genres.add(tag.strip().lower().replace(" ", "_").replace("&", "and"))

    print(f"[INFO] Unique genres found in train: {sorted(all_genres)}")

    # create one binary column per genre
    def encode_genres(df, genres):
        for genre in genres:
            df[f"genre_{genre}"] = df["genre"].apply(
                lambda g: 1 if isinstance(g, str) and genre in
                [t.strip().lower().replace(" ", "_").replace("&", "and") for t in g.split(",")]
                else 0
            )
        return df.drop(columns=["genre"])

    train_final = encode_genres(train_final, all_genres)
    val_final = encode_genres(val_final,   all_genres)
    test_final = encode_genres(test_final,  all_genres)

    # save
    train_final.to_csv(train_out, index=False)
    val_final.to_csv(val_out, index=False)
    test_final.to_csv(test_out, index=False)

    print(f"[INFO] Genre encoded into {len(all_genres)} binary columns and files updated")

    con.close()
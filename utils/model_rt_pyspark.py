import pandas as pd

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, Imputer
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder


BASE_FEATURE_COLS = [
    # "log_numvotes",
    # "director_success_rate",
    "director_movie_count",
    # "writer_success_rate",
    "writer_movie_count",
    "runtimeMinutes",
    "year",
    "title_is_same",
]

RT_NUMERIC_FEATURE_COLS = [
    "tomatoMeter",
    "audienceScore",
    "has_rt_match",
]


def run(
    train_csv: str = "data/rt_train.csv",
    val_csv: str = "data/rt_validation.csv",
    test_csv: str = "data/rt_test.csv",
    val_out: str = "submissions/rt_validation_submission.csv",
    test_out: str = "submissions/rt_test_submission.csv",
) -> None:

    spark = (
        SparkSession.builder
        .master("local[*]")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.driver.memory", "4g")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    train = spark.read.csv(train_csv, header=True, inferSchema=True)
    val = spark.read.csv(val_csv, header=True, inferSchema=True)
    test = spark.read.csv(test_csv, header=True, inferSchema=True)

    label_str = F.lower(F.trim(F.col("label").cast("string")))
    train = train.withColumn(
        "label",
        F.when(label_str.isin("1", "true"), F.lit(1))
         .when(label_str.isin("0", "false"), F.lit(0))
         .otherwise(F.lit(None).cast("int"))
         .cast("int")
    )

    total_train_rows = train.count()
    null_label_rows = train.filter(F.col("label").isNull()).count()
    if null_label_rows > 0:
        print(f"[WARN] Dropping {null_label_rows} train rows with null/invalid labels")
    train = train.filter(F.col("label").isNotNull())
    kept_train_rows = train.count()
    print(f"[INFO] Train rows kept for fitting: {kept_train_rows}/{total_train_rows}")

    # split for local val data, with labels
    train_data, local_val_data = train.randomSplit([0.8, 0.2], seed=42)
    print(f"[INFO] Training on {train_data.count()} rows, Local Val on {local_val_data.count()} rows")

    train_cols = set(train_data.columns)
    val_cols = set(val.columns)
    test_cols = set(test.columns)
    common_cols = train_cols & val_cols & test_cols

    genre_cols = sorted(c for c in common_cols if c.startswith("genre_"))
    feature_cols = [
        c for c in (BASE_FEATURE_COLS + RT_NUMERIC_FEATURE_COLS + genre_cols)
        if c in common_cols
    ]

    if not feature_cols:
        raise ValueError("No common feature columns found across train/val/test.")

    print(f"[INFO] Using {len(feature_cols)} features")
    print(f"[INFO] RT numeric features present: {[c for c in RT_NUMERIC_FEATURE_COLS if c in feature_cols]}")
    print(f"[INFO] Genre one-hot features present: {len(genre_cols)}")

    imputer = Imputer(
        inputCols=feature_cols,
        outputCols=[f"{c}_imp" for c in feature_cols],
        strategy="median",
    )

    imputed_cols = [f"{c}_imp" for c in feature_cols]

    assembler = VectorAssembler(
        inputCols=imputed_cols,
        outputCol="features",
    )

    gbt = GBTClassifier(
        labelCol="label",
        featuresCol="features",
        maxIter=100,    # <- reduce due to compute constraint
        maxDepth=3,     # <- reduce due to compute constraint
        stepSize=0.05,
        subsamplingRate=0.8,
        seed=42,
    )

    pipeline = Pipeline(stages=[imputer, assembler, gbt])

    param_grid = (
        ParamGridBuilder()
        .addGrid(gbt.maxDepth, [3, 4])      # <- reduce due to compute constraint
        .addGrid(gbt.maxIter, [100])        # <- reduce due to compute constraint
        .build()
    )

    evaluator = BinaryClassificationEvaluator(
        labelCol="label",
        metricName="areaUnderROC",
    )

    cv = CrossValidator(
        estimator=pipeline,
        estimatorParamMaps=param_grid,
        evaluator=evaluator,
        numFolds=3,     # <- reduce due to compute constraint
        seed=42,
    )

    print("=" * 60)
    print("TRAINING — 3-fold CV over param grid")
    print("=" * 60)

    cv_model = cv.fit(train_data)
    best_model = cv_model.bestModel

    print(f"[INFO] Best CV AUC-ROC: {max(cv_model.avgMetrics):.4f}")
    print(f"[INFO] All CV AUC-ROC scores: {[round(m, 4) for m in cv_model.avgMetrics]}")

    train_preds = best_model.transform(train_data)
    val_preds = best_model.transform(local_val_data)

    acc_eval = MulticlassClassificationEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName="accuracy",
    )
    print(f"[INFO] Train accuracy: {acc_eval.evaluate(train_preds):.4f}")
    print(f"[INFO] Validation accuracy: {acc_eval.evaluate(val_preds):.4f}")

    imputer_model = best_model.stages[0]
    assembler_model = best_model.stages[1]
    gbt_model = best_model.stages[2]

    val_imputed = imputer_model.transform(local_val_data)
    val_features = assembler_model.transform(val_imputed)
    val_errors = gbt_model.evaluateEachIteration(val_features)

    print(f"[INFO] Validation Loss at Epoch 1:   {val_errors[0]:.4f}")
    print(f"[INFO] Validation Loss at Epoch 50:  {val_errors[len(val_errors)//2]:.4f}")
    print(f"[INFO] Validation Loss at Epoch {len(val_errors)}: {val_errors[-1]:.4f}")

    feature_importance_data = []

    print("\n[INFO] Feature importances:")
    for col, score in sorted(zip(imputed_cols, gbt_model.featureImportances), key=lambda x: -x[1]):
        clean_name = col.replace('_imp', '')
        print(f"  {clean_name:<30} {score:.4f}")
        feature_importance_data.append({"feature": clean_name, "importance": score})

    pd.DataFrame(feature_importance_data).to_csv("data/rt_feature_importance_results.csv", index=False)
    print("[INFO] Feature importances saved to data/rt_feature_importance_results.csv")

    def save_predictions(df, output_path, split_name):
        preds = best_model.transform(df)

        if "tconst" in preds.columns:
            preds = preds.orderBy("tconst")

        output = (
            preds
            .select(
                F.when(F.col("prediction").cast("int") == 1, F.lit("True"))
                 .otherwise(F.lit("False"))
                 .alias("label")
            )
            .toPandas()
        )

        output.to_csv(output_path, index=False, header=False)
        print(f"[INFO] {split_name} predictions saved to: {output_path} ({len(output)} lines)")

    save_predictions(val, val_out, "Validation")
    save_predictions(test, test_out, "Test")

    spark.stop()


if __name__ == "__main__":
    run()

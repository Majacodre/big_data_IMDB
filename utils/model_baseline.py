from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, Imputer
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder


FEATURE_COLS = [
    "log_numvotes",
    "director_success_rate",
    "director_movie_count",
    "writer_success_rate",
    "writer_movie_count",
    "runtimeMinutes",
    "year",
    "title_is_same",
]


def run(
    train_csv: str = "data/features_train.csv",
    val_csv: str = "data/features_validation.csv",
    test_csv: str = "data/features_test.csv",
    val_out: str = "submission/validation_predictions.txt",
    test_out: str = "submission/test_predictions.txt",
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

    train = train.withColumn("label", F.col("label").cast("int"))

    imputer = Imputer(
        inputCols=FEATURE_COLS,
        outputCols=[f"{c}_imp" for c in FEATURE_COLS],
        strategy="median",
    )

    imputed_cols = [f"{c}_imp" for c in FEATURE_COLS]

    assembler = VectorAssembler(
        inputCols=imputed_cols,
        outputCol="features",
    )

    gbt = GBTClassifier(
        labelCol="label",
        featuresCol="features",
        maxIter=300,
        maxDepth=4,
        stepSize=0.05,
        subsamplingRate=0.8,
        seed=42,
    )

    pipeline = Pipeline(stages=[imputer, assembler, gbt])

    param_grid = (
        ParamGridBuilder()
        .addGrid(gbt.maxDepth, [3, 4, 5])
        .addGrid(gbt.maxIter, [200, 300])
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
        numFolds=5,
        seed=42,
    )

    print("=" * 60)
    print("TRAINING — 5-fold CV over param grid")
    print("=" * 60)

    cv_model = cv.fit(train)
    best_model = cv_model.bestModel

    print(f"[INFO] Best CV AUC-ROC: {max(cv_model.avgMetrics):.4f}")
    print(f"[INFO] All CV AUC-ROC scores: {[round(m, 4) for m in cv_model.avgMetrics]}")

    train_preds = best_model.transform(train)
    acc_eval = MulticlassClassificationEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName="accuracy",
    )
    print(f"[INFO] Train accuracy: {acc_eval.evaluate(train_preds):.4f}")

    gbt_stage = best_model.stages[-1]
    print("\n[INFO] Feature importances:")
    for col, score in sorted(zip(imputed_cols, gbt_stage.featureImportances), key=lambda x: -x[1]):
        print(f"  {col.replace('_imp', ''):<30} {score:.4f}")

    print()
    print("=" * 60)
    print("PREDICTIONS")
    print("=" * 60)

    def save_predictions(df, output_path, split_name):
        preds = best_model.transform(df)

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

"""from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, Imputer
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder


# Static feature columns — genre_* columns are added dynamically below
STATIC_FEATURE_COLS = [
    "log_numvotes",
    "director_success_rate",
    "director_movie_count",
    "writer_success_rate",
    "writer_movie_count",
    "runtimeMinutes",
    "year",
    "title_is_same",
    "tomatoMeter",
    "audienceScore",
    "has_rt_match",
]

# column names with special characters that Spark can't handle in VectorAssembler
RENAME_MAP = {
    "genre_lgbtq+": "genre_lgbtq",
    "genre_sci-fi":  "genre_scifi",
}


def run(
    train_csv: str = "data/rt_train.csv",
    val_csv:   str = "data/rt_validation.csv",
    test_csv:  str = "data/rt_test.csv",
    val_out:   str = "submissions/validation_submission.csv",
    test_out:  str = "submissions/test_submission.csv",
) -> None:

    spark = (
        SparkSession.builder
        .master("local[*]")
        .config("spark.driver.memory", "8g")
        .config("spark.executor.memory", "8g")
        .config("spark.sql.shuffle.partitions", "4")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    train = spark.read.csv(train_csv, header=True, inferSchema=True)
    val   = spark.read.csv(val_csv,   header=True, inferSchema=True)
    test  = spark.read.csv(test_csv,  header=True, inferSchema=True)

    train = train.withColumn("label", F.col("label").cast("int"))

    # Rename problematic column names for Spark compatibility
    # genre_lgbtq+ and genre_sci-fi contain special chars that break VectorAssembler
    for old, new in RENAME_MAP.items():
        if old in train.columns:
            train = train.withColumnRenamed(old, new)
            val   = val.withColumnRenamed(old, new)
            test  = test.withColumnRenamed(old, new)

    # Pick up genre_* columns dynamically from train
    genre_cols   = [c for c in train.columns if c.startswith("genre_")]
    FEATURE_COLS = [c for c in STATIC_FEATURE_COLS if c in train.columns] + genre_cols
    imputed_cols = [f"{c}_imp" for c in FEATURE_COLS]

    imputer = Imputer(
        inputCols=FEATURE_COLS,
        outputCols=imputed_cols,
        strategy="median",
    )

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
        .addGrid(gbt.maxDepth, [4, 5])
        .addGrid(gbt.maxIter,  [300])
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

    cv_model   = cv.fit(train)
    best_model = cv_model.bestModel

    print(f"[INFO] Best CV AUC-ROC:       {max(cv_model.avgMetrics):.4f}")
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
        print(f"  {col.replace('_imp', ''):<35} {score:.4f}")

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

    save_predictions(val,  val_out,  "Validation")
    save_predictions(test, test_out, "Test")

    spark.stop()
"""

# migrated to sklearn because of OOM errors with Spark on large datasets
import os
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

STATIC_FEATURE_COLS = [
    "log_numvotes",
    "director_success_rate",
    "director_movie_count",
    "writer_success_rate",
    "writer_movie_count",
    "runtimeMinutes",
    "year",
    "title_is_same",
    "tomatoMeter",
    "audienceScore",
    "has_rt_match",
]

def run(
    train_csv: str = "data/rt_train.csv",
    val_csv:   str = "data/rt_validation.csv",
    test_csv:  str = "data/rt_test.csv",
    val_out:   str = "submissions/validation_submission.csv",
    test_out:  str = "submissions/test_submission.csv",
) -> None:

    os.makedirs("submissions", exist_ok=True)

    train = pd.read_csv(train_csv)
    val   = pd.read_csv(val_csv)
    test  = pd.read_csv(test_csv)

    genre_cols   = [c for c in train.columns if c.startswith("genre_")]
    feature_cols = [c for c in STATIC_FEATURE_COLS if c in train.columns] + genre_cols

    X_train = train[feature_cols].fillna(train[feature_cols].median())
    y_train = train["label"].astype(int)
    X_val   = val[feature_cols].fillna(train[feature_cols].median())
    X_test  = test[feature_cols].fillna(train[feature_cols].median())

    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")
    print(f"[INFO] 5-fold CV accuracy: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

    model.fit(X_train, y_train)
    print(f"[INFO] Train accuracy: {accuracy_score(y_train, model.predict(X_train)):.4f}")

    importances = pd.Series(model.feature_importances_, index=feature_cols)
    print("\n[INFO] Feature importances:")
    for feat, score in importances.sort_values(ascending=False).items():
        print(f"  {feat:<35} {score:.4f}")

    def save(X, output_path, split_name):
        preds = model.predict(X)
        pd.DataFrame({"label": ["True" if p == 1 else "False" for p in preds]}).to_csv(
            output_path, index=False, header=False
        )
        print(f"[INFO] {split_name} predictions saved to: {output_path} ({len(preds)} lines)")

    save(X_val,  val_out,  "Validation")
    save(X_test, test_out, "Test")

if __name__ == "__main__":
    run()
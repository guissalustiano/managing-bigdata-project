from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import LinearSVC, RandomForestClassifier
from pyspark.sql.functions import col, isnan, isnull, when, count, lit, rand, round
import utilities
import functools
import argparse
import json
import sys

session = None

def get_session():
    global session
    if session is None:
        session = SparkSession.builder.getOrCreate()
    return session

def load_data(data_path):
    print("loading data from {}...".format(data_path))
    data = get_session().read.parquet(data_path)
    data = data.withColumn("label", when(data["Label"] == "ddos", 1).otherwise(0))
    print("vectorizing data...")
    data = utilities.vectorize(data)
    print("selecting features...")
    # data = utilities.standard_feature_select(data)
    data = utilities.minmax_feature_select(data)
    return data

def k_fold(data, model_name, folds=5, seed=404):
    print("--- started k-fold validation ---")

    if folds < 2:
        raise RuntimeError("expects at least 2 folds but is given {}".format(folds))

    print("computing splits...")
    splits = data.randomSplit([1.0 for _ in range(folds)], seed=seed)

    metrics = {
        "accuracy": [],
        "weightedFMeasure": [],
        "weightedPrecision": [],
        "weightedRecall": [],
        'precisionByLabel - 0': [],
        'precisionByLabel - 1': [],
        'recallByLabel - 0': [],
        'recallByLabel - 1': [],
        'fMeasureByLabel - 0': [],
        'fMeasureByLabel - 1': []
    }

    for i in range(folds):
        print("------- starting fold {} -------".format(i+1))

        print("spawning new model...")
        model = getattr(sys.modules[__name__], model_name)(featuresCol="features", labelCol="label")

        print("assembling train test splits...")
        train = [splits[k] for k in range(len(splits)) if k != i]
        train = functools.reduce(lambda df1, df2: df1.union(df2.select(df1.columns)), train, train.pop(0))
        test = splits[i]

        print("fitting model...")
        model = model.fit(train)

        print("computing predictions...")
        predictions = model.transform(test)

        print("computing statistics...")
        metrics['accuracy'].append(MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy").evaluate(predictions))
        metrics['weightedFMeasure'].append(MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedFMeasure").evaluate(predictions))
        metrics['weightedPrecision'].append(MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedPrecision").evaluate(predictions))
        metrics['weightedRecall'].append(MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedRecall").evaluate(predictions))
        metrics['precisionByLabel - 0'].append(MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricLabel=0, metricName="precisionByLabel").evaluate(predictions))
        metrics['precisionByLabel - 1'].append(MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricLabel=1, metricName="precisionByLabel").evaluate(predictions))
        metrics['recallByLabel - 0'].append(MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricLabel=0, metricName="recallByLabel").evaluate(predictions))
        metrics['recallByLabel - 1'].append(MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricLabel=1, metricName="recallByLabel").evaluate(predictions))
        metrics['fMeasureByLabel - 0'].append(MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricLabel=0, metricName="fMeasureByLabel").evaluate(predictions))
        metrics['fMeasureByLabel - 1'].append(MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricLabel=1, metricName="fMeasureByLabel").evaluate(predictions))

    print("------------- done ------------")
    return metrics


if __name__ == "__main__":
    # create parser
    parser = argparse.ArgumentParser(prog='machine_learning', description='performs k-fold validation')

    # add command line arguments
    parser.add_argument('--student', default='1999133', type=str, help="exchange to interface with")
    parser.add_argument('--folds', default=5, type=int, help="number of folds")
    parser.add_argument('--seed', default=404, type=int, help="RNG seed")

    # parse command line arguments
    args = parser.parse_args()

    student_id = "s{}".format(args.student)

    data = load_data("/user/{}/final_dataset_parquet_cleaned".format(student_id))

    for model in ['LinearSVC', 'RandomForestClassifier']:
        metrics = k_fold(data, model, args.folds, args.seed)

        print("------------- All runs for {} -------------".format(model))
        print(metrics)

        metrics = {name: sum(value) / args.folds for name, value in metrics.items()}

        print("-------- Averaged metrics for {} ---------".format(model))
        print(metrics)



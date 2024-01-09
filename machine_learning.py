from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import LinearSVC, RandomForestClassifier
from pyspark.sql.functions import col, isnan, isnull, when, count, lit, rand, round
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
    return data


def create_assembler():
    columns = [
        # "Flow ID",
        # "Src IP",
        "Src Port",
        # "Dst IP",
        "Dst Port",
        "Protocol",
        # "Timestamp",
        "Flow Duration",
        "Tot Fwd Pkts",
        "Tot Bwd Pkts",
        "TotLen Fwd Pkts",
        "TotLen Bwd Pkts",
        "Fwd Pkt Len Max",
        "Fwd Pkt Len Min",
        "Fwd Pkt Len Mean",
        "Fwd Pkt Len Std",
        "Bwd Pkt Len Max",
        "Bwd Pkt Len Min",
        "Bwd Pkt Len Mean",
        "Bwd Pkt Len Std",
        "Flow Byts/s",
        "Flow Pkts/s",
        "Flow IAT Mean",
        "Flow IAT Std",
        "Flow IAT Max",
        "Flow IAT Min",
        "Fwd IAT Tot",
        "Fwd IAT Mean",
        "Fwd IAT Std",
        "Fwd IAT Max",
        "Fwd IAT Min",
        "Bwd IAT Tot",
        "Bwd IAT Mean",
        "Bwd IAT Std",
        "Bwd IAT Max",
        "Bwd IAT Min",
        "Fwd Header Len",
        "Bwd Header Len",
        "Fwd Pkts/s",
        "Bwd Pkts/s",
        "Pkt Len Min",
        "Pkt Len Max",
        "Pkt Len Mean",
        "Pkt Len Std",
        "Pkt Len Var",
        "SYN Flag Cnt",
        "RST Flag Cnt",
        "PSH Flag Cnt",
        "ACK Flag Cnt",
        "CWE Flag Count",
        "ECE Flag Cnt",
        "Down/Up Ratio",
        "Pkt Size Avg",
        "Fwd Seg Size Avg",
        "Bwd Seg Size Avg",
        "Subflow Fwd Pkts",
        "Subflow Fwd Byts",
        "Subflow Bwd Pkts",
        "Subflow Bwd Byts",
        "Init Fwd Win Byts",
        "Init Bwd Win Byts",
        "Fwd Act Data Pkts",
        "Fwd Seg Size Min",
        "Active Mean",
        "Active Max",
        "Active Min",
        "Idle Mean",
        "Idle Std",
        "Idle Max",
        "Idle Min",
    ]

    print("creating assembler with columns {}...".format(columns))
    return VectorAssembler(inputCols=columns, outputCol="features")


def k_fold(data, model_name, assembler, folds=5, seed=404):
    print("--- started k-fold validation ---")

    print("vectorizing data...")
    data = assembler.transform(data)

    print("computing splits...")
    splits = data.randomSplit([1.0] + [1.0 for _ in range(folds)], seed=seed)

    metrics = {
        "accuracy": 0.0,
        "weightedFMeasure": 0.0,
        "weightedPrecision": 0.0,
        "weightedRecall": 0.0,
        'precisionByLabel - 0': 0.0,
        'precisionByLabel - 1': 0.0,
        'recallByLabel - 0': 0.0,
        'recallByLabel - 1': 0.0,
        'fMeasureByLabel - 0': 0.0,
        'fMeasureByLabel - 1': 0.0
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
        metrics['accuracy'] += MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy").evaluate(predictions)
        metrics['weightedFMeasure'] += MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedFMeasure").evaluate(predictions)
        metrics['weightedPrecision'] += MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedPrecision").evaluate(predictions)
        metrics['weightedRecall'] += MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedRecall").evaluate(predictions)
        metrics['precisionByLabel - 0'] += MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricLabel=0, metricName="precisionByLabel").evaluate(predictions)
        metrics['precisionByLabel - 1'] += MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricLabel=1, metricName="precisionByLabel").evaluate(predictions)
        metrics['recallByLabel - 0'] += MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricLabel=0, metricName="recallByLabel").evaluate(predictions)
        metrics['recallByLabel - 1'] += MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricLabel=1, metricName="recallByLabel").evaluate(predictions)
        metrics['fMeasureByLabel - 0'] += MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricLabel=0, metricName="fMeasureByLabel").evaluate(predictions)
        metrics['fMeasureByLabel - 1'] += MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricLabel=1, metricName="fMeasureByLabel").evaluate(predictions)

    # average statistics
    metrics = {name: value / folds for name, value in metrics.items()}

    print("------------- done ------------")
    return metrics


if __name__ == "__main__":
    # create parser
    parser = argparse.ArgumentParser(prog='data_preparation', description='prepares data for further processing')

    # add command line arguments
    parser.add_argument('--student', default='1999133', type=str, help="exchange to interface with")
    parser.add_argument('--folds', default=5, type=int, help="number of folds")
    parser.add_argument('--seed', default=404, type=int, help="RNG seed")

    # parse command line arguments
    args = parser.parse_args()

    student_id = "s{}".format(args.student)

    data = load_data("/user/{}/final_dataset_parquet_cleaned".format(student_id))
    assembler = create_assembler()

    for model in ['LinearSVC', 'RandomForestClassifier']:
        metrics = k_fold(data, model, assembler, args.folds, args.seed)
        print("{}: {}".format(model, metrics))

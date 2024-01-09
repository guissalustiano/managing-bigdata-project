from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import LinearSVC
from pyspark.sql.functions import col, isnan, isnull, when, count, lit, rand, round
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
    spark = get_session()
    df = spark.read.parquet('/user/s3301311/final_dataset_parquet')
    # df = spark.read.parquet('/user/s1999133/final_dataset_parquet_cleaned')

    # Convert Label to int
    df = df.replace(float('inf'), None).replace(float('-inf'), None)
    df = df.dropna()
    df = df.withColumn("label", when(df["Label"] == "ddos", 1).otherwise(0))
    # df = df.drop("Label")
    return df

    # print("loading data from {}...".format(data_path))
    # data = spark.read.parquet(data_path)
    # data = data.withColumn("label", when(data["Label"] == "ddos", 1).otherwise(0))
    # return data


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

    print("vectorizing data...")
    data = assembler.transform(data)

    print("computing splits...")
    data = data.withColumn("fold", round(rand(seed) * folds).cast("int"))

    metrics = {
        "accuracy": 0.0,
    }

    for i in range(folds):
        print("starting fold {}...".format(i+1))

        print("spawning new model...")
        model = getattr(sys.modules[__name__], model_name)(featuresCol="features", labelCol="label")

        print("splitting data into train test...")
        train = data.filter(data.fold != i)
        test = data.filter(data.fold == i)

        print("fitting model...")
        model = model.fit(train)

        print("computing predictions...")
        predictions = model.transform(test)

        print("computing statistics...")
        metrics['accuracy'] += MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy").evaluate(predictions)

    # average statistics
    for name, value in metrics.items():
        value /= folds

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

    for model in ['LinearSVC']:
        metrics = k_fold(data, model, assembler, args.folds, args.seed)
        print(metrics)

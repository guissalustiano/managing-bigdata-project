from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import LinearSVC
from pyspark.sql.functions import col, isnan, isnull, when, count, lit
import argparse
import json


def load_data(data_path):
    spark = SparkSession.builder.getOrCreate()
    df = spark.read.parquet('/user/s1999133/final_dataset_parquet')
    # df = spark.read.parquet('/user/s1999133/final_dataset_parquet_cleaned')

    # Convert Label to int
    df = df.replace(float('inf'), None).replace(float('-inf'), None)
    df = df.dropna()
    df = df.withColumn("label", when(df["Label"] == "ddos", 1).otherwise(0))
    return sd


    print("loading data from {}...".format(data_path))
    spark = SparkSession.builder.getOrCreate()
    data = spark.read.parquet(data_path)
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


def k_fold(data, model, assembler, split=[0.8, 0.2], folds=5, seed=404, optimize_params=False):
    print("starting k-fold using {} folds and {} seed...".format(folds, seed))

    param_grid = ParamGridBuilder()

    if optimize_params:
        (param_grid
            .addGrid(model.regParam, [0.1, 0.01])
            .addGrid(model.maxIter, [10, 100]))

    pipeline = Pipeline(stages=[assembler, model])

    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")

    cross_validator = CrossValidator(estimator=pipeline,
                                     estimatorParamMaps=param_grid.build(),
                                     evaluator=evaluator,
                                     numFolds=folds,
                                     seed=seed)

    train, test = data.randomSplit(split, seed=seed)
    print(train.count(), test.count())

    model = cross_validator.fit(train)
    predictions = model.transform(test)

    statistics = {
        "accuracy": MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction",
                                                  metricName="weightedPrecision").evaluate(predictions)
    }

    return statistics


if __name__ == "__main__":
    # create parser
    parser = argparse.ArgumentParser(prog='data_preparation', description='prepares data for further processing')

    # add command line arguments
    parser.add_argument('--student', default='1999133', type=str, help="exchange to interface with")
    parser.add_argument('--folds', default=5, type=int, help="number of folds")
    parser.add_argument('--seed', default=404, type=int, help="RNG seed")
    parser.add_argument('--split', default='[0.8, 0.2]', type=str, help="train-test split")
    parser.add_argument('--optimize', default=False, type=bool, help="Whenever to optimize hyperparameters")

    # parse command line arguments
    args = parser.parse_args()

    student_id = "s{}".format(args.student)

    data = load_data("/user/{}/final_dataset_parquet_cleaned".format(student_id))
    assembler = create_assembler()

    for model in [LinearSVC(featuresCol="features", labelCol="label")]:
        statistics = k_fold(data, model, assembler, json.loads(args.split), args.folds, args.seed, args.optimize)
        print(statistics)

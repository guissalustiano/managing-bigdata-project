#For train and test split

from pyspark.sql import SparkSession
from pyspark.sql.functions import when, col
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LinearSVC
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import BinaryClassificationEvaluator

SEED = 404

spark = SparkSession.builder.getOrCreate()
df = spark.read.parquet('/user/s3301311/final_dataset_parquet')
#df = spark.read.parquet('/user/s1999133/final_dataset_parquet_cleaned')

# Convert Label to int
df = df.replace(float('inf'), None).replace(float('-inf'), None)
df = df.dropna()
df = df.withColumn("label", when(df["Label"] == "ddos", 1).otherwise(0))
#df = df.drop("Label")

inputCols = [
        #"Flow ID",
        #"Src IP",
        "Src Port",
        #"Dst IP",
        "Dst Port",
        "Protocol",
        #"Timestamp",
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

print("slit data")
train_data, test_data = df.randomSplit([0.8, 0.2], seed=SEED)

print("vectorize")
assembler = VectorAssembler(inputCols=inputCols, outputCol="features")
train_data = assembler.transform(train_data)
test_data = assembler.transform(test_data)

svm = LinearSVC(featuresCol="features", labelCol="label")

model = svm.fit(train_data)

predictions = model.transform(test_data)
evaluator = BinaryClassificationEvaluator()
accuracy = evaluator.evaluate(predictions)

# Evaluate the predictions and print the accuracy
# The evaluator uses the default metric (area under ROC curve) to evaluate accuracy
print(f"Test Accuracy: {accuracy}")

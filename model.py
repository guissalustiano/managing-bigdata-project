#For train and test split

from pyspark.sql import SparkSession
from pyspark.sql.functions import when
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LinearSVC
from pyspark.ml.evaluation import BinaryClassificationEvaluator

SEED = 404

spark = SparkSession.builder.getOrCreate()
df = spark.read.parquet('/user/s3301311/final_dataset_parquet')

df = df.sample(fraction=0.05, seed=SEED) # Just for test
df = df.dropna()


# Convert Label to int
df = df.withColumn("IsDDOS", when(df["Label"] == "ddos", 1).otherwise(0))
#df = df.drop("Label")

train_data, test_data = df.randomSplit([0.8, 0.2], seed=SEED)

assembler = VectorAssembler(inputCols=[
#        "Flow ID",
#        "Src IP",
        "Src Port",
#        "Dst IP",
        "Dst Port",
        "Protocol",
#        "Timestamp",
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
        "Fwd PSH Flags",
        "Bwd PSH Flags",
        "Fwd URG Flags",
        "Bwd URG Flags",
        "Fwd Header Len",
        "Bwd Header Len",
        "Fwd Pkts/s",
        "Bwd Pkts/s",
        "Pkt Len Min",
        "Pkt Len Max",
        "Pkt Len Mean",
        "Pkt Len Std",
        "Pkt Len Var",
        "FIN Flag Cnt",
        "SYN Flag Cnt",
        "RST Flag Cnt",
        "PSH Flag Cnt",
        "ACK Flag Cnt",
        "URG Flag Cnt",
        "CWE Flag Count",
        "ECE Flag Cnt",
        "Down/Up Ratio",
        "Pkt Size Avg",
        "Fwd Seg Size Avg",
        "Bwd Seg Size Avg",
        "Fwd Byts/b Avg",
        "Fwd Pkts/b Avg",
        "Fwd Blk Rate Avg",
        "Bwd Byts/b Avg",
        "Bwd Pkts/b Avg",
        "Bwd Blk Rate Avg",
        "Subflow Fwd Pkts",
        "Subflow Fwd Byts",
        "Subflow Bwd Pkts",
        "Subflow Bwd Byts",
        "Init Fwd Win Byts",
        "Init Bwd Win Byts",
        "Fwd Act Data Pkts",
        "Fwd Seg Size Min",
        "Active Mean",
        "Active Std",
        "Active Max",
        "Active Min",
        "Idle Mean",
        "Idle Std",
        "Idle Max",
        "Idle Min",
        "IsDDOS",
    ], 
    outputCol="features"
)
train_data = assembler.transform(train_data)

svm = LinearSVC(featuresCol="features", labelCol="IsDDOS")
model = svm.fit(train_data)

predictions = model.transform(test_data)
evaluator = BinaryClassificationEvaluator()
accuracy = evaluator.evaluate(predictions)

print(f"accuracy: {accuracy}")

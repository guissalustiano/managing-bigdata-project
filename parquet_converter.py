from pyspark.sql import SparkSession

def convert2parquet(filepath: str):
    output = filepath.replace('.csv', '.parquet')

    spark = SparkSession.builder.getOrCreate()
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    df.write.parquet(output, override=True)


if __name__ == "__main__":
    convert2parquet("/user/s3301311/final_dataset.csv")
    convert2parquet("/user/s3301311/unbalaced_20_80_dataset.csv")


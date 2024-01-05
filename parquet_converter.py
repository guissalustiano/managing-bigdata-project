import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, explode, count


if __name__ == "__main__":
    input = "/user/s3301311/final_dataset.csv"
    output = "/user/s3301311/final_dataset.parquet"

    spark = SparkSession.builder.getOrCreate()
    df = spark.read.csv(input, header=True, inferSchema=True)
    df.write.parquet(output)

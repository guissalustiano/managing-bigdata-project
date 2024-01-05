from pyspark.sql import SparkSession

if __name__ == "__main__":
    spark = SparkSession.builder.getOrCreate()
    dataset = spark.read.parquet('/user/s3301311/final_dataset_parquet')


    print(dataset.head(10))


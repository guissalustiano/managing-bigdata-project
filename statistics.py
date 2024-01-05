from pyspark.sql import SparkSession
from pyspark.sql.functions import col, explode, count

if __name__ == "__main__":
    spark = SparkSession.builder.getOrCreate()
    cvs = spark.read.cs


    first = spark.read.json('/data/doina/WebInsight/2020-09-07/*.gz').select(col('url'),
                                                                             col('fetch.contentLength').alias(
                                                                                 'content1'))
    second = spark.read.json('/data/doina/WebInsight/2020-09-14/*.gz').select(col('url'),
                                                                              col('fetch.contentLength').alias(
                                                                                  'content2'))
    filtered = first.join(second, 'url').filter((col("content1") > 0) | (col("content2") > 0))
    result = filtered.withColumn('diff', col('content1') - col('content2')).select(col('url'), col('diff')).sort('diff')
    # result.write.csv("/user/s1999133/WEB/", header=True, mode='overwrite')
    print(result.head(10))


# hdfs dfs -cat /user/s3301311/final_dataset.csv
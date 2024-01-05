from pyspark.sql import SparkSession
from pyspark.sql.functions import col, isnan, when, count, lit
import argparse


def convert2parquet(input: str, output=None):

    if output is None:
        output = input.replace('.csv', '_parquet')

    spark = SparkSession.builder.getOrCreate()
    df = spark.read.csv(input, header=True, inferSchema=True)
    df.write.parquet(output, mode='overwrite')


def clean_data(datapath: str, predominance_threshold=0.95, nan_threshold=0.05):
    spark = SparkSession.builder.getOrCreate()
    df = spark.read.parquet(datapath, header=True, inferSchema=True)

    print("started data cleaning for {}...".format(datapath))

    # map infs to nans
    for col_name in df.columns:
        df = df.withColumn(col_name, when(isnan(1 / col(col_name)), float('nan')).otherwise(col(col_name)))

    print("removed nans...")

    def get_predominant_category(col_name):
        category_counts = df.groupBy(col_name).count()
        total_count = df.count()
        predominant_category = category_counts.orderBy(col("count").desc()).first()
        category_percentage = predominant_category['count'] / total_count
        print(predominant_category['count'], total_count)
        return {
            'col': col_name,
            'unique_values': category_counts.count(),
            'most_frequent_value': predominant_category[col_name],
            'frequency': category_percentage
        }

    # compute predominant values
    predominant_categories = [
        get_predominant_category(col_name)
        for col_name in [col_name for col_name, dtype in df.dtypes]
    ]

    predominant_df = spark.createDataFrame(predominant_categories)

    cols_to_drop = predominant_df.filter(predominant_df['frequency'] >= predominance_threshold).select(
        'col').rdd.flatMap(lambda x: x).collect()

    # drop columns where frequency exceeds threshold
    df_cleaned = df.drop(*cols_to_drop)

    print("dropped unbalanced columns...")

    # compute %nan of all entries
    total_rows = df_cleaned.count()
    nan_percentages = [
        (col_name, df_cleaned.filter(col(col_name).isNull()).count() / total_rows)
        for col_name in df_cleaned.columns
    ]

    columns_to_drop = [col_name for col_name, nan_percentage in nan_percentages if nan_percentage > nan_threshold]

    # drop columns with more than 5%nan values
    df_cleaned = df_cleaned.drop(*columns_to_drop)

    print("dropped columns with too many nans...")

    # drop rows with nans
    df_cleaned = df.dropna()

    print("dropped remaining nans...")

    # save cleaned data
    df_cleaned.write.parquet('{}_cleaned'.format(datapath), mode='overwrite')


if __name__ == "__main__":
    # create parser
    parser = argparse.ArgumentParser(prog='data_preparation', description='prepares data for further processing')

    # add command line arguments
    parser.add_argument('--student', default='3301311', type=str, help="exchange to interface with")

    # parse command line arguments
    args = parser.parse_args()

    student_id = "s{}".format(args.student)

    # convert2parquet("/user/s3301311/final_dataset.csv", "/user/{}/final_dataset_parquet".format(student_id))
    # convert2parquet("/user/s3301311/unbalaced_20_80_dataset.csv", "/user/{}/unbalaced_20_80_dataset_parquet".format(student_id))

    clean_data("/user/{}/final_dataset_parquet".format(student_id))
    # clean_data("/user/{}/unbalaced_20_80_dataset_parquet".format(student_id))

from pyspark.sql import SparkSession
import argparse




if __name__ == "__main__":
    # create parser
    parser = argparse.ArgumentParser(prog='describe_dataset', description='prints dataset rows and columns')

    # add command line arguments
    parser.add_argument('--student', default='1999133', type=str, help="")

    # parse command line arguments
    args = parser.parse_args()

    dataset_path = "/user/{}/final_dataset_parquet_cleaned".format("s{}".format(args.student))

    session = SparkSession.builder.getOrCreate()
    df = session.read.parquet(dataset_path, header=True, inferSchema=True)

    columns = [col_name for col_name, dtype in df.dtypes]
    print("column count: {}, columns: {}".format(len(columns), columns))
    print("row count: {}".format(df.count()))


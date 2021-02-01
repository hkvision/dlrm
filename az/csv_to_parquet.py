from pyspark.sql import SparkSession
from pyspark.sql.types import *
import os

LABEL_COL = 0
INT_COLS = list(range(1, 14))
CAT_COLS = list(range(14, 40))

if __name__ == '__main__':
    spark = SparkSession.builder.getOrCreate()
    base_path = "hdfs://172.16.0.165:9000/dlrm/terabyte"
    out_path = "hdfs://172.16.0.165:9000/dlrm/parquet"
    day_number = 1

    label_fields = [StructField('_c%d' % LABEL_COL, IntegerType())]
    int_fields = [StructField('_c%d' % i, IntegerType()) for i in INT_COLS]
    str_fields = [StructField('_c%d' % i, StringType()) for i in CAT_COLS]

    schema = StructType(label_fields + int_fields + str_fields)
    # paths = [os.path.join(folder, 'day_%d' % i) for i in day_range]
    for file in ["day_23_test", "day_23_validation"]:
        path = os.path.join(base_path, file)
        df = spark.read.schema(schema).option('sep', '\t').csv(path)
        path = os.path.join(out_path, '{}.parquet'.format(file))
        df.write.parquet(path, mode="overwrite")

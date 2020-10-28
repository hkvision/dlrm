import numpy as np
from pyspark.sql import SQLContext
from pyspark.sql.types import *
from pyspark.sql.functions import col, udf, array
from pyspark.ml.feature import StringIndexer
from zoo import init_nncontext

sc = init_nncontext()
sqlContext = SQLContext.getOrCreate(sc)
spark = sqlContext.sparkSession

feature_dicts = []
feature_count = []

label_fields = [StructField("y", IntegerType())]
int_fields = [StructField("_c%d" % i, IntegerType()) for i in list(range(1, 14))]
str_fields = [StructField("_c%d" % i, StringType()) for i in list(range(14, 40))]
int_fields_name = [field.name for field in int_fields]
str_fields_name = [field.name for field in str_fields]

schema = StructType(label_fields + int_fields + str_fields)
df = spark.read.schema(schema).option("sep", "\t").csv("/home/kai/Downloads/dac_sample/dac/train.txt")

fillNA = udf(lambda value: "0" if value == "" or value == "\n" or value is None else value)
convertToInt = udf(lambda value: int(value, 16), IntegerType())
zeroThreshold = udf(lambda value: 0 if int(value) < 0 else value)

for field in str_fields_name:
    df = df.withColumn(field, convertToInt(fillNA(col(field))))
    indexer = StringIndexer(inputCol=field, outputCol="indexed" + field)
    indexer_model = indexer.fit(df)
    df = indexer_model.transform(df)
    index_map = indexer_model.labels
    feature_dicts.append(index_map)  # labels returns the list and the position in the list is the index
    feature_count.append(len(index_map))

for field in int_fields_name:
    df = df.withColumn(field, zeroThreshold(fillNA(col(field))))

int_cols = [col(field) for field in int_fields_name]
str_cols = [col("indexed" + field) for field in str_fields_name]
df = df.withColumn("X_int", array(int_cols))
df = df.withColumn("X_cat", array(str_cols))
df = df.select("y", "X_int", "X_cat")

df.show(5)

rows = df.collect()
X_int = np.array([row[1] for row in rows], dtype=np.int32)
X_cat = np.array([row[2] for row in rows])
y = np.array([row[0] for row in rows], dtype=np.int32)

np.savez_compressed(
            "/home/kai/Downloads/dac_sample/dac/spark_processed.npz",
            X_cat=X_cat,
            X_int=X_int,
            y=y,
            counts=np.array(feature_count, dtype=np.int32),
        )

days = 7
num_data_per_split, extras = divmod(len(rows), days)
total_per_file = [num_data_per_split] * days
for j in range(extras):
    total_per_file[j] += 1

np.savez_compressed("/home/kai/Downloads/dac_sample/dac/zoo/train_day_count.npz",
                    total_per_file=np.array(total_per_file, dtype=np.int32))

sc.stop()

import time
import pickle
import numpy as np
from pyspark.sql import SQLContext
from pyspark.sql.types import *
from pyspark.sql.functions import col, udf, array
from zoo.orca import init_orca_context

hdfs_path = "hdfs://172.16.0.103:9000/dlrm/"
# path = "/opt/work/client/kai/recommendation/kaggle/"
path = "/home/kai/Downloads/dac_sample/dac/zoo/"
sc = init_orca_context(cores=16, memory="20g")
# sc = init_orca_context(cluster_mode="yarn", num_nodes=4, cores=44, memory="80g", driver_memory="80g")
sqlContext = SQLContext.getOrCreate(sc)
spark = sqlContext.sparkSession

label_fields = [StructField("y", IntegerType())]
int_fields = [StructField("_c%d" % i, IntegerType()) for i in list(range(1, 14))]
str_fields = [StructField("_c%d" % i, StringType()) for i in list(range(14, 40))]
int_fields_name = [field.name for field in int_fields]
str_fields_name = [field.name for field in str_fields]

schema = StructType(label_fields + int_fields + str_fields)
start_time = time.time()
df = spark.read.schema(schema).option("sep", "\t").csv(path + "train.txt")
# df = spark.read.schema(schema).option("sep", "\t").csv(hdfs_path + "train.txt")

with open(path + "train_fea_dict.pkl", 'rb') as f:
    feature_dict = pickle.load(f)

def map_to_id(map_broadcast, category):
    return map_broadcast.value[category]


fillNA = udf(lambda value: "0" if value == "" or value == "\n" or value is None else value)
convertToInt = udf(lambda value: int(value, 16), IntegerType())
zeroThreshold = udf(lambda value: 0 if int(value) < 0 else value)

for field in str_fields_name:
    feature_dict_broadcast = sc.broadcast(feature_dict[field])
    categorify = udf(lambda value: map_to_id(feature_dict_broadcast, value))
    df = df.withColumn(field, fillNA(col(field)))
    df = df.withColumn(field, categorify(col(field)))

for field in int_fields_name:
    df = df.withColumn(field, zeroThreshold(fillNA(col(field))))

df.show(5)

int_cols = [col(field) for field in int_fields_name]
str_cols = [col(field) for field in str_fields_name]
df = df.withColumn("X_int", array(int_cols))
df = df.withColumn("X_cat", array(str_cols))
df = df.select("y", "X_int", "X_cat")

df.show(5)

total = df.count()
print(total)


def split(total_size, num_parts):
    part_size, extras = divmod(total_size, num_parts)
    res = [part_size] * num_parts
    for j in range(extras):
        res[j] += 1
    return res


days = 7
total_per_file = split(total, days)
print(total_per_file)

with np.load(path + "train_fea_count.npz") as data:
    counts = data["counts"]


# rows = df.rdd.collect()
# X_int = np.array([row[1] for row in rows], dtype=np.int32)
# X_cat = np.array([row[2] for row in rows])
# y = np.array([row[0] for row in rows], dtype=np.int32)
#
# np.savez_compressed(
#             path + "spark_processed.npz",
#             X_cat=X_cat,
#             X_int=X_int,
#             y=y,
#             counts=counts)

def save(path):
    def save_func(index, iterator):
        data = list(iterator)
        X_int = np.array([row[1] for row in data], dtype=np.int32)
        X_cat = np.array([row[2] for row in data])
        y = np.array([row[0] for row in data], dtype=np.int32)
        import pyarrow as pa
        fs = pa.hdfs.connect()
        with fs.open(path + "spark_processed_" + str(index) + ".npz", 'wb') as f:
            np.savez_compressed(
                        f,
                        X_cat=X_cat,
                        X_int=X_int,
                        y=y,
                        counts=counts)
        yield 0

    return save_func

df.rdd.mapPartitionsWithIndex(save(path)).collect()
# df.rdd.mapPartitionsWithIndex(save(hdfs_path)).collect()


np.savez_compressed(path + "train_day_count.npz",
                    total_per_file=np.array(total_per_file, dtype=np.int32))


end_time = time.time()
print("Time used: ", end_time - start_time)
sc.stop()

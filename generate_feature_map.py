import time
import pickle
import numpy as np
from pyspark.sql import SQLContext
from pyspark.sql.types import *
from pyspark.sql.functions import col, udf
from pyspark.ml.feature import StringIndexer
from zoo.orca import init_orca_context

hdfs_path = "hdfs://172.16.0.103:9000/dlrm/"
# path = "/opt/work/client/kai/recommendation/kaggle/"
path = "/home/kai/Downloads/dac_sample/dac/zoo/"
sc = init_orca_context(cores=16, memory="20g")
# sc = init_orca_context(cluster_mode="yarn", num_nodes=4, cores=44, memory="80g", driver_memory="80g")
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
start_time = time.time()
df = spark.read.schema(schema).option("sep", "\t").csv(path + "train.txt")
# df = spark.read.schema(schema).option("sep", "\t").csv(hdfs_path + "train.txt")

fillNA = udf(lambda value: "0" if value == "" or value == "\n" or value is None else value)
# str -> int -> stringindexer -> int would cause incompatibility?
convertToInt = udf(lambda value: int(value, 16), IntegerType())
zeroThreshold = udf(lambda value: 0 if int(value) < 0 else value)

for field in str_fields_name:
    df = df.withColumn(field, fillNA(col(field)))
    indexer = StringIndexer(inputCol=field, outputCol="indexed" + field)
    indexer_model = indexer.fit(df)
    df = indexer_model.transform(df)
    index_map = indexer_model.labels
    feature_dicts.append(index_map)  # labels returns the list and the position in the list is the index
    feature_count.append(len(index_map))

feature_map = {}
for i in range(0, len(feature_dicts)):
    field_name = str_fields_name[i]
    feature_i_map = {}
    feature_list = feature_dicts[i]
    for i, item in enumerate(feature_list):
        feature_i_map[item] = i
    feature_map[field_name] = feature_i_map


with open(path + "train_fea_dict.pkl", "wb") as f:
    pickle.dump(feature_map, f, pickle.HIGHEST_PROTOCOL)

np.savez_compressed(path + "train_fea_count.npz",
                    counts=np.array(feature_count, dtype=np.int32))


end_time = time.time()
print("Time used: ", end_time - start_time)
sc.stop()

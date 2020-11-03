
#!/bin/bash

# Copyright (c) 2020 NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#########################################################################
# File Name: run_spark.sh

set -e

# the environment variables to run spark job
# should modify below environment variables

# the data path including 1TB criteo data, day_0, day_1, ...
export INPUT_PATH=/dlrm/input
#export INPUT_PATH=/media/criteo/

# the output path, use for generating the dictionary and the final dataset
# the output folder should have more than 300GB
export OUTPUT_PATH=/dlrm/output

# spark local dir should have about 3TB
# the temporary path used for spark shuffle write
export SPARK_LOCAL_DIRS=/mnt/disk1/,/mnt/disk2/,/mnt/disk3/

# below numbers should be adjusted according to the resource of your running environment
# set the total number of CPU cores, spark can use
export TOTAL_CORES=336

# set the number of executors
export NUM_EXECUTORS=42

# the cores for each executor, it'll be calculated
export NUM_EXECUTOR_CORES=$((${TOTAL_CORES}/${NUM_EXECUTORS}))

# unit: GB,  set the max memory you want to use
export TOTAL_MEMORY=1000

# unit: GB, set the memory for driver
export DRIVER_MEMORY=32

# the memory per executor
#export EXECUTOR_MEMORY=$(((${TOTAL_MEMORY}-${DRIVER_MEMORY})/${NUM_EXECUTORS}))
export EXECUTOR_MEMORY=45

# use frequency_limit=15 or not
# by default use a frequency limit of 15
USE_FREQUENCY_LIMIT=1
OPTS=""
if [[ $USE_FREQUENCY_LIMIT == 1 ]]; then
    OPTS="--frequency_limit 15"
fi

export JAVA_HOME=/opt/jdk1.8.0_232-cloudera
export SPARK_HOME=/opt/spark-3.0.1-bin-hadoop277
export PATH=$JAVA_HOME/bin:$SPARK_HOME/bin:$PATH


# we use spark standalone to run the job
export MASTER=spark://clx1:7077

temp_test=hdfs://clx1:8020/dlrm/input/test
temp_validation=hdfs://clx1:8020/dlrm/input/validation


#temp_test=/media/criteo/test
#temp_validation=/media/criteo/validation


spark-submit --master $MASTER \
        --driver-memory "${DRIVER_MEMORY}G" \
        --executor-cores $NUM_EXECUTOR_CORES \
        --executor-memory "${EXECUTOR_MEMORY}G" \
        --conf spark.cores.max=$TOTAL_CORES \
        --conf spark.task.cpus=1 \
        --conf spark.sql.files.maxPartitionBytes=1105199104 \
        --conf spark.sql.shuffle.partitions=672 \
        --conf spark.driver.maxResultSize=2G \
        --conf spark.locality.wait=0s \
        --conf spark.network.timeout=1800s \
        spark_data_utils.py \
        $OPTS \
        --input_folder $INPUT_PATH \
        --test_input_folder $temp_test \
        --validation_input_folder $temp_validation \
        --output_folder $OUTPUT_PATH/ \
        --model_size_file $OUTPUT_PATH/model_size.json \
        --days 0-23 \
        --train_days 0-22 \
        --remain_days 23-23 \
        --model_folder $OUTPUT_PATH/models\
        --write_mode overwrite --low_mem 2>&1 | tee -a ./submit_total_log.txt

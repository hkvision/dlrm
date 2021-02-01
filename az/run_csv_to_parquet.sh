#!/bin/bash

set -e

spark-submit --master yarn --deploy-mode client \
        --driver-memory 32g \
        --executor-cores 44 \
        --executor-memory 80g \
        --num-executors 4 \
        csv_to_parquet.py
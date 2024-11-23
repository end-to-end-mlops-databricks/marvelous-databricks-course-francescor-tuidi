# Databricks notebook source
# MAGIC %pip install /Volumes/marvelous_dev_ops/video_games_sales/dist/mlops_with_databricks-0.0.1-py3-none-any.whl --force-reinstall --quiet

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import time

import requests
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
    EndpointCoreConfigInput,
    ServedEntityInput,
    TrafficConfig,
    Route,
)

from src import config
from pyspark.sql import SparkSession

# COMMAND ----------

workspace = WorkspaceClient()
spark = SparkSession.builder.getOrCreate()

catalog_name = config.catalog_name
schema_name = config.schema_name

train_set = spark.table(f"{catalog_name}.{schema_name}.train_set").toPandas()
entity_version = 4


workspace.serving_endpoints.create(
    name="video-games-model-serving",
    config=EndpointCoreConfigInput(
        served_entities=[
            ServedEntityInput(
                entity_name=f"{catalog_name}.{schema_name}.video_games_model",
                scale_to_zero_enabled=True,
                workload_size="Small",
                entity_version=entity_version,
            )
        ],
    # Optional if only 1 entity is served
    traffic_config=TrafficConfig(
        routes=[
            Route(served_model_name=f"video_games_model-{entity_version}",
                  traffic_percentage=100)
        ]
        ),
    ),
)


# COMMAND ----------

# MAGIC %md
# MAGIC ## Call the endpoint

# COMMAND ----------

token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
host = spark.conf.get("spark.databricks.workspaceUrl")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create sample request body

# COMMAND ----------

required_columns = config.cat_features + config.num_features

sampled_records = train_set[required_columns].sample(n=1000, replace=True).to_dict(orient="records")
dataframe_records = [[record] for record in sampled_records] # <- list of list of json

"""
Each body should be list of json with columns

[{
    'key1': 'value1',
    'key2': 'value2',
    'key3': 'value3'
}]
"""

# COMMAND ----------

start_time = time.time()

model_serving_endpoint = (
    f"https://{host}/serving-endpoints/video-games-model-serving/invocations"
)
response = requests.post(
    f"{model_serving_endpoint}",
    headers={"Authorization": f"Bearer {token}"},
    json={"dataframe_records": dataframe_records[0]},
)

end_time = time.time()
execution_time = end_time - start_time

print("Response status:", response.status_code)
print("Reponse text:", response.text)
print("Execution time:", execution_time, "seconds")


# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Test

# COMMAND ----------

# Initialize variables
model_serving_endpoint = (
    f"https://{host}/serving-endpoints/video-games-model-serving/invocations"
)

headers = {"Authorization": f"Bearer {token}"}
num_requests = 100


# Function to make a request and record latency
def send_request():
    random_record = random.choice(dataframe_records)
    start_time = time.time()
    response = requests.post(
        model_serving_endpoint,
        headers=headers,
        json={"dataframe_records": random_record},
    )
    end_time = time.time()
    latency = end_time - start_time
    return response.status_code, latency


total_start_time = time.time()
latencies = []

# Send requests concurrently
with ThreadPoolExecutor(max_workers=100) as executor:
    futures = [executor.submit(send_request) for _ in range(num_requests)]

    for future in as_completed(futures):
        status_code, latency = future.result()
        latencies.append(latency)

total_end_time = time.time()
total_execution_time = total_end_time - total_start_time

# Calculate the average latency
average_latency = sum(latencies) / len(latencies)

print("\nTotal execution time:", total_execution_time, "seconds")
print("Average latency per request:", average_latency, "seconds")


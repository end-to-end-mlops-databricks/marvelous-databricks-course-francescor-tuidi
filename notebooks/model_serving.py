# Databricks notebook source
# MAGIC %pip install /Volumes/marvelous_dev_ops/video_games_sales/dist/mlops_with_databricks-0.0.1-py3-none-any.whl --force-reinstall --quiet # noqa

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
    EndpointCoreConfigInput,
    Route,
    ServedEntityInput,
    TrafficConfig,
)

# from pyspark import dbutils
from pyspark.sql import SparkSession

from src import config
from src.utils.requests import send_request, send_request_concurrently

# COMMAND ----------

workspace = WorkspaceClient()
spark = SparkSession.builder.getOrCreate()

# COMMAND ----------

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
            routes=[Route(served_model_name=f"video_games_model-{entity_version}", traffic_percentage=100)]
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
dataframe_records = [[record] for record in sampled_records]  # <- list of list of json

"""
Each body should be list of json with columns

[{
    'key1': 'value1',
    'key2': 'value2',
    'key3': 'value3'
}]
"""

# COMMAND ----------

model_serving_endpoint = f"https://{host}/serving-endpoints/video-games-model-serving/invocations"
headers = {"Authorization": f"Bearer {token}"}

response, status_code, latency = send_request(
    records=dataframe_records[0],
    endpoint=model_serving_endpoint,
    headers=headers,
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Test

# COMMAND ----------

send_request_concurrently(num_requests=2, records=dataframe_records, endpoint=model_serving_endpoint, headers=headers)

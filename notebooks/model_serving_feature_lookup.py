# Databricks notebook source
# MAGIC %pip install /Volumes/marvelous_dev_ops/video_games_sales/dist/mlops_with_databricks-0.0.1-py3-none-any.whl --force-reinstall --quiet # noqa
# MAGIC %pip install databricks-sdk==0.32.0 --force-reinstall

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import time

import requests
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import (
    OnlineTableSpec,
    OnlineTableSpecTriggeredSchedulingPolicy,
)
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput
from pyspark.sql import SparkSession

from src import config
from src.utils.requests import send_request

# COMMAND ----------

spark = SparkSession.builder.getOrCreate()
workspace = WorkspaceClient()

catalog_name = config.catalog_name
schema_name = config.schema_name

# COMMAND ----------

online_table_name = f"{catalog_name}.{schema_name}.video_games_online"
spec = OnlineTableSpec(
    primary_key_columns=["Rank"],
    source_table_full_name=f"{catalog_name}.{schema_name}.video_games_features",
    run_triggered=OnlineTableSpecTriggeredSchedulingPolicy.from_dict({"triggered": "true"}),
    perform_full_copy=False,
)

online_table_pipeline = workspace.online_tables.create(name=online_table_name, spec=spec)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create endpoint

# COMMAND ----------

workspace.serving_endpoints.create(
    name="video-games-model-serving-fe",
    config=EndpointCoreConfigInput(
        served_entities=[
            ServedEntityInput(
                entity_name=f"{catalog_name}.{schema_name}.video_games_model_fe",
                scale_to_zero_enabled=True,
                workload_size="Small",
                entity_version=1,
            )
        ]
    ),
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Call the endpoint

# COMMAND ----------

token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
host = spark.conf.get("spark.databricks.workspaceUrl")

# COMMAND ----------

# Excluding "OverallQual", "GrLivArea", "GarageCars" because they will be taken from feature look up

features_look_up = ["NA_Sales", "JP_Sales", "Other_Sales", "Global_Sales"]
required_columns = config.cat_features + config.num_features
[required_columns.remove(feat) for feat in features_look_up]

train_set = spark.table(f"{catalog_name}.{schema_name}.train_set").toPandas()
train_set["Rank"] = train_set["Rank"].astype(str)
train_set["Year"] = train_set["Year"].astype("float32")

sampled_records = (
    train_set[required_columns].sample(n=1000, replace=True).to_dict(orient="records")
)
dataframe_records = [[record] for record in sampled_records]

# COMMAND ----------

model_serving_endpoint = (
    f"https://{host}/serving-endpoints/video-games-model-serving-fe/invocations"
)

# COMMAND ----------

pred, status_code, latency = send_request(
    records=dataframe_records[0],
    endpoint=model_serving_endpoint,
    headers={"Authorization": f"Bearer {token}"},
)

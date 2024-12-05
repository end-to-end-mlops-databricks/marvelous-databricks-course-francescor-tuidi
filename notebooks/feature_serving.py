# Databricks notebook source
# MAGIC %pip install /Volumes/marvelous_dev_ops/video_games_sales/dist/mlops_with_databricks-0.0.1-py3-none-any.whl --force-reinstall --quiet # noqa
# MAGIC %pip install databricks-sdk==0.32.0 --force-reinstall --quiet

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

"""
Create feature table in unity catalog, it will be a delta table
Create online table which uses the feature delta table created in the previous step
Create a feature spec. When you create a feature spec,
you specify the source Delta table.
This allows the feature spec to be used in both offline and online scenarios.
For online lookups, the serving endpoint automatically uses the online table to perform low-latency feature lookups.
The source Delta table and the online table must use the same primary key.

"""


import mlflow
import pandas as pd
from databricks import feature_engineering
from databricks.feature_engineering import FeatureLookup
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import OnlineTableSpec, OnlineTableSpecTriggeredSchedulingPolicy
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput
from pyspark.sql import SparkSession

from src import config
from src.utils.requests import send_request

spark = SparkSession.builder.getOrCreate()

# Initialize Databricks clients
workspace = WorkspaceClient()
fe = feature_engineering.FeatureEngineeringClient()

# Set the MLflow registry URI
mlflow.set_registry_uri("databricks-uc")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load config, train and test tables

# COMMAND ----------

# Get feature columns details
num_features = config.num_features
cat_features = config.cat_features
target = config.target
catalog_name = config.catalog_name
schema_name = config.schema_name

# Define table names
feature_table_name = f"{catalog_name}.{schema_name}.video_games_preds"
online_table_name = f"{catalog_name}.{schema_name}.video_games_preds_online"

# Load training and test sets from Catalog
train_set = spark.table(f"{catalog_name}.{schema_name}.train_set").toPandas()
test_set = spark.table(f"{catalog_name}.{schema_name}.test_set").toPandas()

df = pd.concat([train_set, test_set])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load a registered model

# COMMAND ----------

# Load the MLflow model for predictions
pipeline = mlflow.sklearn.load_model(f"models:/{catalog_name}.{schema_name}.video_games_model/4")

# COMMAND ----------

# Prepare the DataFrame for predictions and feature table creation - these features are the ones we want to serve.
preds_df = df[["Rank", "Genre", "Global_Sales"]]
preds_df["Predicted_Sales"] = pipeline.predict(df[cat_features + num_features])

preds_df = spark.createDataFrame(preds_df)

# 1. Create the feature table in Databricks

fe.create_table(
    name=feature_table_name, primary_keys=["Rank"], df=preds_df, description="Video games predictions feature table"
)

# Enable Change Data Feed
spark.sql(f"""
    ALTER TABLE {feature_table_name}
    SET TBLPROPERTIES (delta.enableChangeDataFeed = true)
""")

# COMMAND ----------

# 2. Create the online table using feature table

spec = OnlineTableSpec(
    primary_key_columns=["Rank"],
    source_table_full_name=feature_table_name,
    run_triggered=OnlineTableSpecTriggeredSchedulingPolicy.from_dict({"triggered": "true"}),
    perform_full_copy=False,
)

# Create the online table in Databricks
online_table_pipeline = workspace.online_tables.create(name=online_table_name, spec=spec)

# COMMAND ----------

# 3. Create feture look up and feature spec table feature table

# Define features to look up from the feature table
features = [
    FeatureLookup(
        table_name=feature_table_name, lookup_key="Rank", feature_names=["Genre", "Global_Sales", "Predicted_Sales"]
    )
]

# Create the feature spec for serving
feature_spec_name = f"{catalog_name}.{schema_name}.return_predictions"

fe.create_feature_spec(name=feature_spec_name, features=features, exclude_columns=None)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Deploy Feature Serving Endpoint

# COMMAND ----------

# 4. Create endpoing using feature spec

workspace.serving_endpoints.create(
    name="video-games-feature-serving",
    config=EndpointCoreConfigInput(
        served_entities=[
            ServedEntityInput(
                entity_name=feature_spec_name,  # feature spec name defined in the previous step
                scale_to_zero_enabled=True,
                workload_size="Small",  # Define the workload size (Small, Medium, Large)
            )
        ]
    ),
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Call The Endpoint

# COMMAND ----------

token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
host = spark.conf.get("spark.databricks.workspaceUrl")

# COMMAND ----------

serving_endpoint = f"https://{host}/serving-endpoints/video-games-feature-serving/invocations"
send_request(endpoint=serving_endpoint, headers={"Authorization": f"Bearer {token}"}, records=[{"Rank": "100"}])

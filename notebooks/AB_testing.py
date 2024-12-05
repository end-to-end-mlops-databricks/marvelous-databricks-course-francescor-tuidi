# Databricks notebook source
# MAGIC %pip install /Volumes/marvelous_dev_ops/video_games_sales/dist/mlops_with_databricks-0.0.1-py3-none-any.whl --force-reinstall --quiet # noqa
# MAGIC %pip install databricks-sdk==0.32.0 --force-reinstall --quiet

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

import mlflow
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput
from mlflow import MlflowClient
from pyspark.sql import SparkSession

from src import config, logger
from src.utils.requests import send_request
from src.video_game_sales.data_processor import DataProcessor
from src.video_game_sales.models.video_game import VideoGameModel
from src.video_game_sales.models.video_game_wrapper_AB_test import VideoGamesModelWrapperABTest

# COMMAND ----------

# Set up MLflow for tracking and model registry
mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")

# Initialize the MLflow client for model management
client = MlflowClient()

# Extract key configuration details
num_features = config.num_features
cat_features = config.cat_features
target = config.target
catalog_name = config.catalog_name
schema_name = config.schema_name
ab_test_params = config.ab_test

# Common Data Processor
data_processor = DataProcessor(config=config)

# COMMAND ----------

# Set up specific parameters for model A and model B as part of the A/B test
parameters_a = {
    "learning_rate": ab_test_params["learning_rate_a"],
}

parameters_b = {
    "learning_rate": ab_test_params["learning_rate_b"],
}

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Load and Prepare Training and Testing Datasets

# COMMAND ----------

spark = SparkSession.builder.getOrCreate()

# Load training and testing sets from Databricks tables
train_set_spark, _ = data_processor.load_from_catalog_spark(spark=spark)
train_set, test_set = data_processor.load_from_catalog_pandas(spark=spark)

train_set["Year"] = train_set["Year"].replace("N/A", None)
test_set["Year"] = test_set["Year"].replace("N/A", None)

# Now, safely drop NaN values
train_set.dropna(inplace=True)
test_set.dropna(inplace=True)

# Convert 'Year' column to float
train_set["Year"] = train_set["Year"].astype("float64")
test_set["Year"] = test_set["Year"].astype("float64")

X_train = train_set[num_features + cat_features]
y_train = train_set[target]

X_test = test_set[num_features + cat_features]
y_test = test_set[target]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train Model A and Log with MLflow

# COMMAND ----------

# Overwrite the default configuration with specific parameters for model A
config.parameters = parameters_a
model_a = VideoGameModel(preprocessor=data_processor, config=config)

model_a.run_experiment(
    spark=spark,
    experiment_name="/Shared/video-games-ab",
    model_name="video_games_model_ab",
    X_train=X_train,
    train_set_spark=train_set_spark,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    model_class_str="A",
    register_model=True,
)
model_uri = model_a.set_model_alias(client, "model_A")
model_A = mlflow.sklearn.load_model(model_uri)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train Model B and Log with MLflow

# COMMAND ----------

# Overwrite the default configuration with specific parameters for model A
config.parameters = parameters_b
model_b = VideoGameModel(preprocessor=data_processor, config=config)

model_b.run_experiment(
    spark=spark,
    experiment_name="/Shared/video-games-ab",
    model_name="video_games_model_ab",
    X_train=X_train,
    train_set_spark=train_set_spark,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    model_class_str="B",
    register_model=True,
)
model_uri = model_a.set_model_alias(client, "model_B")
model_B = mlflow.sklearn.load_model(model_uri)

# COMMAND ----------

# MAGIC %md
# MAGIC # Define Custom A/B Test Model

# COMMAND ----------

wrapped_model = VideoGamesModelWrapperABTest(model_a=model_A, model_b=model_B)
example_input = X_test.iloc[0:1]  # Select the first row for prediction as example
example_prediction = wrapped_model.predict(context=None, model_input=example_input)
logger.info(f"Example Prediction: {example_prediction}")


# COMMAND ----------

wrapped_model_name = "video_games_model_pyfunc_ab_test"
run_id, model_version = wrapped_model.run_experiments(
    spark=spark,
    X_train=X_train,
    train_set_spark=train_set_spark,
    experiment_name="/Shared/video-games-ab-testing",
    model_name=wrapped_model_name,
    register_model=True,
)

# COMMAND ----------

model = mlflow.pyfunc.load_model(model_uri=f"models:/{catalog_name}.{schema_name}.{wrapped_model_name}/{model_version}")
predictions = model.predict(X_test.iloc[0:1])
# Display predictions
logger.info(predictions)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create serving endpoint

# COMMAND ----------

workspace = WorkspaceClient()
end_point_name = "video-games-model-serving-ab-test-2"

workspace.serving_endpoints.create(
    name=end_point_name,
    config=EndpointCoreConfigInput(
        served_entities=[
            ServedEntityInput(
                entity_name=f"{catalog_name}.{schema_name}.{wrapped_model_name}",
                scale_to_zero_enabled=True,
                workload_size="Small",
                entity_version=model_version,
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

required_columns = config.cat_features + config.num_features

sampled_records = train_set[required_columns].sample(n=1000, replace=True).to_dict(orient="records")
dataframe_records = [[record] for record in sampled_records]

model_serving_endpoint = f"https://{host}/serving-endpoints/{end_point_name}/invocations"
headers = {"Authorization": f"Bearer {token}"}
res, status_code, latency = send_request(endpoint=model_serving_endpoint, records=dataframe_records, headers=headers)

logger.info(res)

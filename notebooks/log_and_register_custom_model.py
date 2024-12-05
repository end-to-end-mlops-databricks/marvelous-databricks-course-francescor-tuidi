# Databricks notebook source

import os

import mlflow
from mlflow import MlflowClient
from mlflow.models import infer_signature
from mlflow.utils.environment import _mlflow_conda_env
from pyspark.sql import SparkSession

from src import config, logger
from src.utils.delta import get_latest_delta_version
from src.utils.git import get_git_info
from src.video_game_sales.data_processor import DataProcessor
from src.video_game_sales.models.video_game import VideoGameModel
from src.video_game_sales.models.video_game_wrapper import VideoGamesModelWrapper

mlflow.set_registry_uri("databricks-uc")
mlflow.set_tracking_uri("databricks")
client = MlflowClient()

# COMMAND ----------

# Extract configuration details
video_game_model = VideoGameModel(config=config)
data_processor = DataProcessor(config=config)
processing_pipeline = data_processor.create_preprocessing_pipeline()

num_features = config.num_features
cat_features = config.cat_features
target = config.target
catalog_name = config.catalog_name
schema_name = config.schema_name

# COMMAND ----------

spark = SparkSession.builder.getOrCreate()

# Load training and testing sets from Databricks tables
train_set_spark, _ = data_processor.load_from_catalog_spark(spark=spark)
train_set, test_set = data_processor.load_from_catalog_pandas(spark=spark)

X_train = train_set[num_features + cat_features]
y_train = train_set[target]

X_test = test_set[num_features + cat_features]
y_test = test_set[target]
parameters = config.parameters


# COMMAND ----------

experiments_name = os.path.join("/Shared", f"{config.catalog_name}-{config.schema_name}")
model_experiments_name = os.path.join("/Shared", f"{config.catalog_name}-{config.schema_name}_wrapper")
run_id = mlflow.search_runs(
    experiment_names=[experiments_name],
    filter_string="tags.branch='latest'",
).run_id[0]

model = mlflow.sklearn.load_model(f"runs:/{run_id}/lightgbm-pipeline-model")

# COMMAND ----------

wrapped_model = VideoGamesModelWrapper(model)  # we pass the loaded model to the wrapper
example_input = X_test.iloc[0:1]  # Select the first row for prediction as example
example_prediction = wrapped_model.predict(context=None, model_input=example_input)
logger.info(f"Example Prediction: {example_prediction}")

# COMMAND ----------

# this is a trick with custom packages
# https://docs.databricks.com/en/machine-learning/model-serving/private-libraries-model-serving.html
# but does not work with pyspark, so we have a better option :-)

mlflow.set_experiment(experiment_name=model_experiments_name)
git_sha, current_branch = get_git_info()

with mlflow.start_run(tags={"branch": current_branch, "git_sha": git_sha}) as run:
    run_id = run.info.run_id
    signature = infer_signature(model_input=X_train, model_output=example_prediction)
    table_name = f"{catalog_name}.{schema_name}.train_set"
    version = get_latest_delta_version(table_name, spark)
    dataset = mlflow.data.from_spark(train_set_spark, table_name=table_name, version=version)
    mlflow.log_input(dataset, context="training")
    dist_path = f"""
    /{config.volumes_root}/{config.catalog_name}/{config.schema_path}/dist/mlops_with_databricks-0.0.1-py3-none-any.whl
    """
    conda_env = _mlflow_conda_env(
        additional_conda_deps=None,
        additional_pip_deps=[
            dist_path,
        ],
        additional_conda_channels=None,
    )
    mlflow.pyfunc.log_model(
        python_model=wrapped_model,
        artifact_path="pyfunc-video-games-model",
        code_path=[dist_path],
        signature=signature,
    )

# COMMAND ----------

loaded_model = mlflow.pyfunc.load_model(f"runs:/{run_id}/pyfunc-video-games-model")
loaded_model.unwrap_python_model()

# COMMAND ----------

model_name = f"{catalog_name}.{schema_name}.video_games_model_pyfunc"

model_version = mlflow.register_model(
    model_uri=f"runs:/{run_id}/pyfunc-video-games-model", name=model_name, tags={"git_sha": git_sha}
)

# COMMAND ----------

model_version_alias = "current_version"
client.set_registered_model_alias(model_name, model_version_alias, "1")

model_uri = f"models:/{model_name}@{model_version_alias}"
model = mlflow.pyfunc.load_model(model_uri)

# COMMAND ----------

client.get_model_version_by_alias(model_name, model_version_alias)

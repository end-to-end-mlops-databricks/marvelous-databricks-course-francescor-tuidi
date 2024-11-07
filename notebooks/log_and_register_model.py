# Databricks notebook source
import os

import git
import mlflow
from mlflow.models import infer_signature
from pyspark.sql import SparkSession
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src import config, logger
from src.utilies.delta import get_latest_delta_version
from src.video_game_sales.data_processor import DataProcessor
from src.video_game_sales.video_game_model import VideoGameModel

mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")

# COMMAND ----------


# Extract configuration details

video_game_model = VideoGameModel(config=config)
data_processor = DataProcessor(config=config)

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
# Define the preprocessor for categorical features
preprocessor = ColumnTransformer(
    transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)], remainder="passthrough"
)

# Create the pipeline with preprocessing and the LightGBM regressor
pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("regressor", video_game_model.model)])


# COMMAND ----------
experiments_name = os.path.join("/Shared", f"{config.catalog_name}-{config.schema_name}")
mlflow.set_experiment(experiment_name=experiments_name)
repo = git.Repo(search_parent_directories=True)
git_sha = repo.head.object.hexsha
current_branch = repo.active_branch.name

# Start an MLflow run to track the training process
with mlflow.start_run(
    tags={"git_sha": f"{git_sha}", "branch": current_branch},
) as run:
    run_id = run.info.run_id

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    # Evaluate the model performance
    evaluation_metrics = video_game_model.evaluate(y_test, y_pred)
    logger.info(f"Evaluation metrics: {evaluation_metrics}")

    # Log parameters, metrics, and the model to MLflow
    mlflow.log_param("model_type", "LightGBM with preprocessing")
    mlflow.log_params(parameters)
    mlflow.log_metric("mse", evaluation_metrics.mse)
    mlflow.log_metric("mae", evaluation_metrics.mae)
    mlflow.log_metric("r2_score", evaluation_metrics.r2)
    signature = infer_signature(model_input=X_train, model_output=y_pred)

    table_name = f"{catalog_name}.{schema_name}.train_set"
    version = get_latest_delta_version(table_name, spark)
    dataset = mlflow.data.from_spark(train_set_spark, table_name=table_name, version=version)
    mlflow.log_input(dataset, context="training")
    mlflow.sklearn.log_model(sk_model=pipeline, artifact_path="lightgbm-pipeline-model", signature=signature)


# COMMAND ----------
model_version = mlflow.register_model(
    model_uri=f"runs:/{run_id}/lightgbm-pipeline-model",
    name=f"{catalog_name}.{schema_name}.house_prices_model_basic",
    tags={"git_sha": f"{git_sha}"},
)

# COMMAND ----------
run = mlflow.get_run(run_id)
dataset_info = run.inputs.dataset_inputs[0].dataset
dataset_source = mlflow.data.get_source(dataset_info)
dataset_source.load()

# COMMAND ----------

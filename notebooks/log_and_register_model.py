# Databricks notebook source
import os

import mlflow
from mlflow.models import infer_signature
from pyspark.sql import SparkSession
from sklearn.pipeline import Pipeline

from src import config, logger
from src.utils.delta import get_latest_delta_version
from src.utils.experiments_runner import ExperimentsRunner
from src.utils.git import get_git_info
from src.video_game_sales.data_processor import DataProcessor
from src.video_game_sales.models.video_game import VideoGameModel

mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")

# COMMAND ----------

video_game_model = VideoGameModel(config=config)
data_processor = DataProcessor(config=config)

# Extract configuration details
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

train_set['Year'] = train_set['Year'].replace('N/A', None)
test_set['Year'] = test_set['Year'].replace('N/A', None)

# Now, safely drop NaN values
train_set.dropna(inplace=True)
test_set.dropna(inplace=True)

# Convert 'Year' column to float
train_set['Year'] = train_set['Year'].astype('float64')
test_set['Year'] = test_set['Year'].astype('float64')

# Continue with your code
X_train = train_set[num_features + cat_features]
y_train = train_set[target]

X_test = test_set[num_features + cat_features]
y_test = test_set[target]
parameters = config.parameters

# COMMAND ----------

# Create the pipeline with preprocessing and the LightGBM regressor
pipeline = Pipeline(steps=[("preprocessor", data_processor.preprocessor), ("regressor", video_game_model.model)])

# COMMAND ----------

git_sha, current_branch = get_git_info()

# COMMAND ----------

data_processor.load_data("/" + config.data_full_path)
data_processor.preprocess_data()
train_set, test_set  = data_processor.split_data()

_X_train = train_set[num_features + cat_features]

# COMMAND ----------

experiments_name = os.path.join("/Shared", f"{config.catalog_name}-{config.schema_name}")
mlflow.set_experiment(experiment_name=experiments_name)

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
    mlflow.log_metric("r2_score", evaluation_metrics.r2_score)
    signature = infer_signature(model_input=X_train, model_output=y_pred)

    table_name = f"{catalog_name}.{schema_name}.train_set"
    version = get_latest_delta_version(table_name, spark)
    dataset = mlflow.data.from_spark(train_set_spark, table_name=table_name, version=version)
    mlflow.log_input(dataset, context="training")
    mlflow.sklearn.log_model(sk_model=pipeline, artifact_path="lightgbm-pipeline-model", signature=signature)


# COMMAND ----------

model_version = mlflow.register_model(
    model_uri=f"runs:/{run_id}/lightgbm-pipeline-model",
    name=f"{catalog_name}.{schema_name}.video_games_model",
    tags={"git_sha": f"{git_sha}"},
)

# COMMAND ----------

run = mlflow.get_run(run_id)
dataset_info = run.inputs.dataset_inputs[0].dataset
dataset_source = mlflow.data.get_source(dataset_info)
dataset_source.load()

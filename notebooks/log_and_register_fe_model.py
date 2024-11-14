# Databricks notebook source

import os
from datetime import datetime

import mlflow
from databricks import feature_engineering
from databricks.feature_engineering import FeatureFunction, FeatureLookup
from databricks.sdk import WorkspaceClient
from mlflow.models import infer_signature
from pyspark.sql import SparkSession
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

from src import config
from src.utils.git import get_git_info
from src.video_game_sales.data_processor import DataProcessor
from src.video_game_sales.models.video_game import VideoGameModel

# COMMAND ----------

# Initialize the Databricks session and clients
processor = DataProcessor()
video_game_model = VideoGameModel()
preprocessing_pipeline = processor.create_preprocessing_pipeline()
spark = SparkSession.builder.getOrCreate()

workspace = WorkspaceClient()
fe = feature_engineering.FeatureEngineeringClient()

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")
mlflow.set_tracking_uri("databricks")

# Extract configuration details
num_features = config.num_features
cat_features = config.cat_features
target = config.target
parameters = config.parameters
catalog_name = config.catalog_name
schema_name = config.schema_name

# Define table names and function name
feature_table_name = f"{catalog_name}.{schema_name}.video_games_features"
function_output_name = "mean_sales"
function_name = f"{catalog_name}.{schema_name}.{function_output_name}"
train_set_path = f"{catalog_name}.{schema_name}.train_set"
test_set_path = f"{catalog_name}.{schema_name}.test_set"


# COMMAND ----------

train_set = spark.table(train_set_path)
test_set = spark.table(test_set_path)

# COMMAND ----------

processor.create_mean_feature_table(spark=spark, function_name=function_name, feature_table_name=feature_table_name)

# COMMAND ----------

features_cols = ["NA_Sales", "JP_Sales", "Other_Sales", "Global_Sales"]
train_set = spark.table(f"{catalog_name}.{schema_name}.train_set").drop(
    "NA_Sales", "JP_Sales", "Other_Sales", "Global_Sales"
)
test_set = spark.table(f"{catalog_name}.{schema_name}.test_set").toPandas()
train_set = train_set.withColumn("Rank", train_set["Rank"].cast("string"))

# COMMAND ----------

# Feature engineering setup
training_set = fe.create_training_set(
    df=train_set,
    label=target,
    feature_lookups=[
        FeatureLookup(
            table_name=feature_table_name,
            feature_names=features_cols,
            lookup_key="Rank",
        ),
        FeatureFunction(
            udf_name=function_name,
            output_name=function_output_name,
            input_bindings={
                "na_sales": "NA_Sales",
                "jp_sales": "JP_Sales",
                "other_sales": "Other_Sales",
                "global_sales": "Global_Sales",
            },
        ),
    ],
    exclude_columns=["update_timestamp_utc"],
)

# COMMAND ----------

# Load feature-engineered DataFrame
train_set = training_set.load_df().toPandas()

# Calculate house_age for training and test set
current_year = datetime.now().year
test_set[function_output_name] = test_set[features_cols].mean(axis=1)

train_set["Year"] = train_set["Year"].replace("N/A", None)
test_set["Year"] = test_set["Year"].replace("N/A", None)
train_set.dropna(inplace=True)
test_set.dropna(inplace=True)
train_set["Year"] = train_set["Year"].astype("float64")
test_set["Year"] = test_set["Year"].astype("float64")

# Split features and target
X_train = train_set[num_features + cat_features + [function_output_name]]
y_train = train_set[target]
X_test = test_set[num_features + cat_features + [function_output_name]]
y_test = test_set[target]

# Setup preprocessing and model pipeline
pipeline = Pipeline(steps=[("preprocessor", preprocessing_pipeline), ("regressor", video_game_model.model)])

# COMMAND ----------

# Set and start MLflow experiment
model_experiments_name = os.path.join("/Shared", f"{config.catalog_name}-{config.schema_name}_fe")
mlflow.set_experiment(experiment_name=model_experiments_name)
git_sha, current_branch = get_git_info()

with mlflow.start_run(tags={"branch": current_branch, "git_sha": git_sha}) as run:
    run_id = run.info.run_id
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    # Calculate and print metrics
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")
    print(f"Mean Absolute Error: {mae}")
    print(f"R2 Score: {r2}")

    # Log model parameters, metrics, and model
    mlflow.log_param("model_type", "LightGBM with preprocessing")
    mlflow.log_params(parameters)
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("r2_score", r2)
    signature = infer_signature(model_input=X_train, model_output=y_pred)

    # Log model with feature engineering
    fe.log_model(
        model=pipeline,
        flavor=mlflow.sklearn,
        artifact_path="lightgbm-pipeline-model-fe",
        training_set=training_set,
        signature=signature,
    )
mlflow.register_model(
    model_uri=f"runs:/{run_id}/lightgbm-pipeline-model-fe", name=f"{catalog_name}.{schema_name}.video_games_model_fe"
)

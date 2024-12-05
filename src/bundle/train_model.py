"""
This script trains a LightGBM model for prediction with feature engineering.
Key functionality:
- Loads training and test data from Databricks tables
- Performs feature engineering using Databricks Feature Store
- Creates a pipeline with preprocessing and LightGBM regressor
- Tracks the experiment using MLflow
- Logs model metrics, parameters and artifacts
- Handles feature lookups and custom feature functions
- Outputs model URI for downstream tasks

The model uses both numerical and categorical features, including a custom calculated feature.
"""

import argparse

import mlflow
from databricks import feature_engineering
from databricks.feature_engineering import FeatureFunction, FeatureLookup
from databricks.sdk import WorkspaceClient
from pyspark.sql import SparkSession

from src import config
from src.video_game_sales.data_processor import DataProcessor
from src.video_game_sales.models.video_game import VideoGameModel

# Job Parameters
parser = argparse.ArgumentParser()
parser.add_argument(
    "--root_path",
    action="store",
    default=None,
    type=str,
    required=True,
)
parser.add_argument(
    "--git_sha",
    action="store",
    default=None,
    type=str,
    required=True,
)
parser.add_argument(
    "--job_run_id",
    action="store",
    default=None,
    type=str,
    required=True,
)
args = parser.parse_args()
root_path = args.root_path
git_sha = args.git_sha
job_run_id = args.job_run_id
data_processor = DataProcessor(config=config)

# Initialize the Databricks session and clients
spark: SparkSession = SparkSession.builder.getOrCreate()
workspace = WorkspaceClient()
fe = feature_engineering.FeatureEngineeringClient()

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

# Load training and test sets
features = ["NA_Sales", "JP_Sales", "Other_Sales", "Global_Sales"]
train_set = spark.table(f"{catalog_name}.{schema_name}.train_set").drop(*features)
test_set = spark.table(f"{catalog_name}.{schema_name}.test_set").toPandas()

# Feature engineering setup
training_set = fe.create_training_set(
    df=train_set,
    label=target,
    feature_lookups=[
        FeatureLookup(
            table_name=feature_table_name,
            feature_names=features,
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

train_set = training_set.load_df().toPandas()

# Calculate house_age for training and test set
test_set[function_output_name] = test_set[features].mean(axis=1)

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
processor = DataProcessor()
model = VideoGameModel(preprocessor=data_processor, config=config)

run_id, model_version = model.run_experiment(
    spark=spark,
    experiment_name="/Shared/video-games-ab",
    model_name="video_games_model_ab",
    X_train=X_train,
    train_set_spark=spark.createDataFrame(train_set),
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    model_class_str="A",
    register_model=True,
)

model_uri = f"runs:/{run_id}/lightgbm-pipeline-model-fe"
dbutils.jobs.taskValues.set(key="new_model_uri", value=model_uri)

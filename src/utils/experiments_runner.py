"""Mlflow experiments runner module."""

import os

import mlflow
from mlflow import MlflowClient
from mlflow.models import infer_signature
from pyspark.sql import SparkSession
from sklearn.pipeline import Pipeline

from src import config as default_config
from src import logger
from src.utils.delta import get_latest_delta_version
from src.utils.git import get_git_info
from src.video_game_sales.config import ProjectConfig
from src.video_game_sales.data_processor import DataProcessor
from src.video_game_sales.models.video_game import VideoGameModel
from src.video_game_sales.models.video_game_wrapper import VideoGamesModelWrapper


class ExperimentsRunner:
    def __init__(
        self,
        model: VideoGameModel | VideoGamesModelWrapper,
        processor: DataProcessor,
        config: ProjectConfig = None,
        experiment_name: str = "experiment",
    ):
        self.config = config or default_config
        self.model = model
        self.processor = processor
        self.experiment_name = experiment_name

    def __get_pipeline(self):
        if self.model and self.processor:
            return Pipeline(steps=[("preprocessor", self.preprocessor), ("regressor", self.model.model)])

    def __get_data(self, spark: SparkSession):
        # Load training and testing sets from Databricks tables
        train_set_spark, _ = self.processor.load_from_catalog_spark(spark=spark)
        train_set, test_set = self.processor.load_from_catalog_pandas(spark=spark)

        train_set, test_set = self.processor.split_data()

        num_features = self.config.num_features
        cat_features = self.config.cat_features
        target = self.config.target

        X_train = train_set[num_features + cat_features]
        y_train = train_set[target]

        X_test = test_set[num_features + cat_features]
        y_test = test_set[target]

        return X_train, train_set_spark, y_train, X_test, y_test

    def run(self, model, model_path):
        mlflow.set_tracking_uri("databricks")
        mlflow.set_registry_uri("databricks-uc")
        client = MlflowClient()

        self.processor.load_data("/" + self.config.data_full_path)
        self.processor.preprocess_data()

        catalog_name = self.config.catalog_name
        schema_name = self.config.schema_name
        parameters = self.config.parameters

        git_sha, current_branch = get_git_info()

        spark = SparkSession.builder.getOrCreate()
        X_train, train_set_spark, y_train, X_test, y_test = self.__get_data(spark)

        experiments_name = os.path.join(
            "/Shared", f"{self.config.catalog_name}-{self.config.schema_name}-{self.experiment_name}"
        )
        mlflow.set_experiment(experiment_name=experiments_name)

        # Start an MLflow run to track the training process

        with mlflow.start_run(
            tags={"git_sha": f"{git_sha}", "branch": current_branch},
        ) as run:
            pipeline = self.__get_pipeline()
            run_id = run.info.run_id

            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)

            # Evaluate the model performance
            evaluation_metrics = self.model.evaluate(y_test, y_pred)
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

            model_version = mlflow.register_model(
                model_uri=f"runs:/{run_id}/lightgbm-pipeline-model",
                name=f"{catalog_name}.{schema_name}.video_games_model",
                tags={"git_sha": f"{git_sha}"},
            )
            logger.info(f"Model version: {model_version}")

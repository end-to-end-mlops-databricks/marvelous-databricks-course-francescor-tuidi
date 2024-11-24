"""This module contains the VideoGameModel class."""

import mlflow
import pandas as pd
from lightgbm import LGBMRegressor
from mlflow import MlflowClient
from mlflow.models import infer_signature
from pyspark.sql import SparkSession
from pyspark.sql.dataframe import DataFrame
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

from src import config as default_config
from src.utils.decorators import log_execution_time
from src.utils.delta import get_latest_delta_version
from src.utils.git import get_git_info
from src.video_game_sales.config import ProjectConfig
from src.video_game_sales.data_processor import DataProcessor
from src.video_game_sales.metrics import Metrics


class VideoGameModel:
    """A model for predicting video game sales."""

    @log_execution_time("Init VideoGameModel.")
    def __init__(self, preprocessor: DataProcessor = None, config: ProjectConfig = None) -> None:
        """Initializes the VideoGameModel object.

        Args:
            config (ProjectConfig): The project configuration.
        """
        self.config = config or default_config
        self.parameters = self.config.parameters
        if preprocessor:
            self.preprocessor = preprocessor
            self.preprocessing_pipeline = self.preprocessor.create_preprocessing_pipeline()
            self.model = LGBMRegressor(**self.parameters)
            self.pipeline = Pipeline(
                steps=[
                    ("preprocessor", self.preprocessing_pipeline),
                    ("regressor", self.model),
                ]
            )

    @log_execution_time("Train model.")
    def train(self, x_train: pd.DataFrame, y_train: pd.DataFrame) -> None:
        """Trains the model.

        Args:
            x_train (pd.DataFrame): The input features.
            y_train (pd.DataFrame): The target variable.
        """
        self.model.fit(x_train, y_train)

    @log_execution_time("Predict.")
    def predict(self, x: pd.DataFrame):
        """Predicts the target variable.

        Args:
            x (pd.DataFrame): The input features.
        """
        self.model.predict(x)

    @log_execution_time("Evaluate.")
    def evaluate(self, y_gt: pd.DataFrame, y_pred: pd.DataFrame) -> Metrics:
        """Evaluates the model.
        Args:
            y_gt (pd.DataFrame): _description_
            y_pred (pd.DataFrame): _description_
        Returns:
            dict: A dictionary containing the evaluation metrics: mse, mae, and r2_score.
        """
        return Metrics.from_dict(
            metrics_dict={
                "mse": mean_squared_error(y_gt, y_pred),
                "mae": mean_absolute_error(y_gt, y_pred),
                "r2_score": r2_score(y_gt, y_pred),
            }
        )

    @log_execution_time("Mlflow Experiment.")
    def run_experiment(
        self,
        spark: SparkSession,
        experiment_name: str,
        model_name: str,
        X_train: pd.DataFrame,
        train_set_spark: DataFrame,
        y_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_test: pd.DataFrame,
        artifact_path: str = "lightgbm-pipeline-model",
        pipeline: Pipeline = None,
        model_class_str: str = None,
        register_model: bool = False,
    ) -> tuple[str, str]:
        """Runs an MLflow experiment.

        Args:
            experiment_name (str): The name of the MLflow experiment.
            model_name (str): The name of the MLflow model.
            pipeline (Pipeline): The pipeline object. If none is provided, the model
                will be created from the class attributes.
            X_train (pd.DataFrame): The training set.
            y_train (pd.DataFrame): The training labels.
            X_test (pd.DataFrame): The test set.
            y_test (pd.DataFrame): The test labels.
            model_class (str): The model class name.
            register_model (bool): Whether to register the model in MLflow.

        Returns:
            tuple(str, str): A tuple containing the run ID and model version.
                The model version is None if the model is not registered.
        """
        pipeline = pipeline or self.pipeline

        mlflow.set_experiment(experiment_name=experiment_name)
        model_name = f"{self.config.catalog_name}.{self.config.schema_name}.{model_name}"
        self.latest_run_model_name = model_name
        git_sha, current_branch = get_git_info()
        tags = {"git_sha": git_sha, "branch": current_branch}

        run_id: str = ""
        model_version: str = None

        if model_class_str:
            tags["model_class"] = model_class_str

        with mlflow.start_run(tags=tags) as run:
            run_id = run.info.run_id
            self.latest_run_id = run_id

            # Train the model
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)

            # Calculate performance metrics
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            # Log model parameters, metrics, and other artifacts in MLflow
            mlflow.log_param("model_type", "LightGBM with preprocessing")
            mlflow.log_params(self.parameters)
            mlflow.log_metric("mse", mse)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("r2_score", r2)
            signature = infer_signature(model_input=X_train, model_output=y_pred)
            table_name = f"{self.config.catalog_name}.{self.config.schema_name}.train_set"
            version_delta = get_latest_delta_version(table_path=table_name, spark=spark)
            # Log the input dataset for tracking reproducibility
            dataset = mlflow.data.from_spark(
                train_set_spark,
                table_name=table_name,
                version=version_delta,
            )
            mlflow.log_input(dataset, context="training")

            # Log the pipeline model in MLflow with a unique artifact path
            mlflow.sklearn.log_model(
                sk_model=pipeline,
                artifact_path=artifact_path,
                signature=signature,
            )
        if register_model:
            model_version = mlflow.register_model(
                model_uri=f"runs:/{run_id}/{artifact_path}",
                name=model_name,
                tags={"git_sha": f"{git_sha}"},
            )
            self.latest_run_model_version = model_version.version
        return run_id, model_version

    @log_execution_time("Set Model Alias.")
    def set_model_alias(
        self, client: MlflowClient, model_alias: str, model_name: str = None, model_version: str = None
    ):
        """Sets the model alias for the latest run.

        Args:
            client (MlflowClient): The MLflow client.
            model_alias (str): The model alias.
            model_name (str): The model name. Defaults to None.
            model_version (str): The model version. Defaults to None.

        Returns:
            str: The model URI of the new model alias.
        """
        model_name = model_name or self.latest_run_model_name
        model_version = model_version or self.latest_run_model_version
        client.set_registered_model_alias(model_name, model_alias, f"{model_version}")
        model_uri = f"models:/{model_name}@{model_alias}"
        return model_uri

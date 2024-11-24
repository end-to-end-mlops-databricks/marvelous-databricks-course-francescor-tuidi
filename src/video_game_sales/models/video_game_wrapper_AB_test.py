import mlflow
import hashlib
import pandas as pd
import mlflow
from src import config as default_confing
from src.utils.decorators import log_execution_time
from src.utils.delta import get_latest_delta_version
from pyspark.sql import SparkSession
from pyspark.sql.dataframe import DataFrame
from src.video_game_sales.config import ProjectConfig
from mlflow.models import infer_signature
from src.utils.git import get_git_info


class VideoGamesModelWrapperABTest(mlflow.pyfunc.PythonModel):
    @log_execution_time("Init VideoGamesModelWrapperABTest.")
    def __init__(self, model_a, model_b, config: ProjectConfig = None) -> None:
        """Initializes the VideoGamesModelWrapperABTest class.

        Args:
            model_a: A MLflow model for the A variant.
            model_b: A MLflow model for the B variant.
        """
        self.config = config or default_confing
        self.model_a = model_a
        self.model_b = model_b

    @log_execution_time("Predict.")
    def predict(self, context, model_input) -> pd.Series | pd.DataFrame:
        """Predicts the price of a video game based on the input.

        Args:
            model_input (pd.DataFrame): A pandas DataFrame containing the input data.

        Returns:
            (pd.DataFrame | pd.Series): A pandas Series containing the predicted price of the video game.
        """

        if isinstance(model_input, pd.DataFrame):
            id = str(model_input["Rank"].values[0])
            hashed_id = hashlib.md5(id.encode(encoding="UTF-8")).hexdigest()
            # convert a hexadecimal (base-16) string into an integer
            if int(hashed_id, 16) % 2:
                predictions = self.model_a.predict(model_input)
                return {"Prediction": predictions[0], "model": "Model A"}
            else:
                predictions = self.model_b.predict(model_input)
                return {"Prediction": predictions[0], "model": "Model B"}
            return predictions
        else:
            raise ValueError("Input must be a pandas DataFrame.")

    @log_execution_time("Run experiments.")
    def run_experiments(
        self,
        spark: SparkSession,
        X_train: pd.DataFrame,
        train_set_spark: DataFrame,
        experiment_name: str,
        model_name="",
        register_model: bool = False,
    ) -> tuple[str, str]:
        """Runs experiments for the VideoGamesModelWrapperABTest class.

        Args:
            spark (SparkSession): A SparkSession object.
            X_train (pd.DataFrame): A pandas DataFrame containing the training data.
            train_set_spark (DataFrame): A Spark DataFrame containing the training data.
            experiment_name (str): The name of the experiment.
            model_name (str): The name of the model.

        Returns:
            run_id (str): The ID of the run.
            model_version (str): The version of the model.
        """
        mlflow.set_experiment(experiment_name=experiment_name)
        model_name = (
            f"{self.config.catalog_name}.{self.config.schema_name}.{model_name}"
        )
        run_id = ""
        model_version = None

        with mlflow.start_run() as run:
            run_id = run.info.run_id
            signature = infer_signature(
                model_input=X_train,
                model_output={"Prediction": 1234.5, "model": "Model B"},
            )
            table_name = (
                f"{self.config.catalog_name}.{self.config.schema_name}.train_set"
            )
            version_delta = get_latest_delta_version(table_path=table_name, spark=spark)
            dataset = mlflow.data.from_spark(
                train_set_spark, table_name=table_name, version=version_delta
            )
            mlflow.log_input(dataset, context="training")
            mlflow.pyfunc.log_model(
                python_model=self,
                artifact_path="pyfunc-video-games-model-ab",
                signature=signature,
            )
        git_sha, current_branch = get_git_info()
        if register_model:
            model_version = mlflow.register_model(
                model_uri=f"runs:/{run_id}/pyfunc-video-games-model-ab",
                name=model_name,
                tags={"git_sha": f"{git_sha}", "branch": f"{current_branch}"},
            ).version
        return run_id, model_version
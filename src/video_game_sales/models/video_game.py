"""This module contains the VideoGameModel class."""

import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src import config as default_config
from src.utils.decorators import log_execution_time
from src.video_game_sales.config import ProjectConfig
from src.video_game_sales.metrics import Metrics


class VideoGameModel:
    """A model for predicting video game sales."""

    @log_execution_time("Init VideoGameModel.")
    def __init__(self, config: ProjectConfig = None) -> None:
        """Initializes the VideoGameModel object.

        Args:
            config (ProjectConfig): The project configuration.
        """
        self.config = config or default_config
        self.parameters = self.config.parameters
        self.model = LGBMRegressor(**self.parameters)

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

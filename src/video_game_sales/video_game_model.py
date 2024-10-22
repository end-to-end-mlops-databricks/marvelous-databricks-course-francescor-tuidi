"""This module contains the VideoGameModel class."""

import pandas as pd

from src.video_game_sales.data_processor import DataProcessor


class VideoGameModel:
    """A model for predicting video game sales."""

    def __init__(self, preprocessor: DataProcessor, config: dict):
        """Initializes the VideoGameModel object.

        Args:
            preprocessor (): _description_
            config (dict): _description_
        """

    def train(self, x_train: pd.DataFrame, y_train: pd.DataFrame):
        """Trains the model.

        Args:
            x_train (pd.DataFrame): _description_
            y_train (pd.DataFrame): _description_
        """

    def predict(self, x: pd.DataFrame):
        """Predicts the target variable.

        Args:
            X (pd.DataFrame): _description_
        """

    def evaluate(self, x_test: pd.DataFrame, y_test: pd.DataFrame):
        """Evaluates the model.

        Args:
            X_test (pd.DataFrame): _description_
            y_test (pd.DataFrame): _description_
        """

    def get_feature_importance(self):
        """Returns the feature importance of the model."""

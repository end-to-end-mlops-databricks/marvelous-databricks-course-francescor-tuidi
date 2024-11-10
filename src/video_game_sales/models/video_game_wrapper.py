import mlflow
import pandas as pd

from src.utils.predictions import adjust_predictions


class VideoGamesModelWrapper(mlflow.pyfunc.PythonModel):
    """A wrapper for the VideoGameModel class."""

    def __init__(self, model) -> None:
        """Initializes the VideoGamesModelWrapper object.

        Args:
            model: The trained model.
        """
        self.model = model

    def predict(self, context, model_input):
        """Predicts the target variable."""
        if isinstance(model_input, pd.DataFrame):
            predictions = self.model.predict(model_input)
            predictions = {"Prediction": adjust_predictions(predictions[0])}
            return predictions
        else:
            raise ValueError("Input must be a pandas DataFrame.")

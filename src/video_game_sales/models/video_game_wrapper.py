import mlflow
import pandas as pd

from src.utils.decorators import log_execution_time
from src.utils.predictions import adjust_predictions


class VideoGamesModelWrapper(mlflow.pyfunc.PythonModel):
    """A wrapper for the VideoGameModel class."""

    @log_execution_time("Init VideoGamesModelWrapper.")
    def __init__(self, model) -> None:
        """Initializes the VideoGamesModelWrapper object.

        Args:
            model: The trained model.
        """
        self.model = model

    @log_execution_time("Predict.")
    def predict(self, context, model_input):
        """Predicts the target variable."""
        if isinstance(model_input, pd.DataFrame):
            predictions = self.model.predict(model_input)
            str_predictions = str(adjust_predictions(predictions[0]))
            predictions = {"Prediction": str_predictions}
            return predictions
        else:
            raise ValueError("Input must be a pandas DataFrame.")

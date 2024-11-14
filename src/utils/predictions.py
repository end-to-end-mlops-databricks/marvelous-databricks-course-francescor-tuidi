"""This module contains utility functions for the predictions phase."""


def adjust_predictions(predictions: float, scale_factor: float = 1.3) -> float:
    """Adjust the predictions by a scale factor.

    Args:
        predictions (float): The predictions to adjust.
        scale_factor (float, optional): The scale factor to use. Defaults to 1.3.
    Returns:
        float: The adjusted predictions.
    """
    return predictions * scale_factor

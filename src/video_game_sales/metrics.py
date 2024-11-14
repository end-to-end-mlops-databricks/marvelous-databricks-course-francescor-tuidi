"""Metrics for the model."""

from pydantic import BaseModel


class Metrics(BaseModel):
    """Metrics for the model."""

    mse: float | list
    mae: float | list
    r2_score: float | list

    @classmethod
    def from_dict(cls, metrics_dict: dict):
        """Load configuration from a YAML file."""
        return cls(**metrics_dict)

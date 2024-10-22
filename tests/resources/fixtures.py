import os

import pandas as pd
import pytest

from src.video_game_sales.data_processor import DataProcessor


@pytest.fixture
def data():
    return pd.DataFrame(
        {
            "num1": [1, 2, 3, 4, 5],
            "num2": [5, 4, 3, 2, 1],
            "cat1": ["A", "B", "C", "A", "B"],
            "cat2": ["X", "Y", "Z", "X", "Y"],
            "target": [10, 20, 30, 40, 50],
        }
    )


@pytest.fixture
def config(tmp_path):
    return {
        "data_path": os.path.join(tmp_path, "data.csv"),
        "num_features": ["num1", "num2"],
        "cat_features": ["cat1", "cat2"],
        "target": "target",
    }


@pytest.fixture
def data_processor(data, config):
    data.to_csv(config["data_path"], index=False)
    return DataProcessor(config)

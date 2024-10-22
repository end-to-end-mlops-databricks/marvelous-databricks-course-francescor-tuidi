"""Tests for the DataProcessor class."""

import numpy as np

from src.video_game_sales.data_processor import DataProcessor


def test_load_data(data_processor, data):
    assert data_processor.df.equals(data)


def test_preprocess_data(data_processor):
    data_processor.preprocess_data()

    assert data_processor.X.shape == (5, 4)
    assert data_processor.y.shape == (5,)
    assert set(data_processor.X.columns) == set(["num1", "num2", "cat1", "cat2"])
    assert data_processor.preprocessor is not None


def test_split_data(data_processor):
    data_processor.preprocess_data()
    X_train, X_test, y_train, y_test = data_processor.split_data(test_size=0.4, random_state=42)

    assert X_train.shape == (3, 4)
    assert X_test.shape == (2, 4)
    assert y_train.shape == (3,)
    assert y_test.shape == (2,)


def test_preprocessor_transform(data_processor):
    data_processor.preprocess_data()
    X_transformed = data_processor.preprocessor.fit_transform(data_processor.X)

    assert X_transformed.shape[0] == 5
    assert X_transformed.shape[1] > 4  # Due to one-hot encoding, we expect more columns


def test_missing_target(tmp_path, data, config):
    data.loc[2, "target"] = np.nan
    data.to_csv(config["data_path"], index=False)
    processor = DataProcessor(config)
    processor.preprocess_data()

    assert processor.df.shape[0] == 4  # One row should be removed due to missing target
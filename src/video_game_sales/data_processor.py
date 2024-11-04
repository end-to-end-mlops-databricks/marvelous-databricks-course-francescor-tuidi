"""This module contains a class for processing video game sales data."""

import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import current_timestamp, to_utc_timestamp
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import logger
from src.utilies.decorators import log_execution_time
from src.video_game_sales.config import ProjectConfig


class DataProcessor:
    """
    A class for processing video game sales data.
    This class handles loading, preprocessing, and splitting the data
        for machine learning tasks.
    """

    @log_execution_time("Initialize DataProcessor.")
    def __init__(self, config: ProjectConfig):
        """
        Initialize the DataProcessor.

        Args:
            filepath (str): Path to the CSV file containing the data.
            config (dict): Configuration dictionary containing feature
                and target information.
        """
        self.df = self.load_data(config.data_path)
        self.config = config
        self.X = None
        self.y = None
        self.preprocessor = None

    @log_execution_time("Load data.")
    def load_data(self, filepath: str):
        """
        Load data from a CSV file.

        Args:
            filepath (str): Path to the CSV file.

        Returns:
            pandas.DataFrame: Loaded data.
        """
        return pd.read_csv(filepath)

    @log_execution_time("Data preprocessing.")
    def preprocess_data(self):
        """
        Preprocess the data by aggregating sales columns, removing rows with
        missing target, separating features and target, and creating
        preprocessing pipelines.
        """

        # Remove rows with missing target
        target = self.config.target

        self.df = self.df.dropna(subset=[target])

        # Separate features and target
        self.X = self.df[self.config.num_features + self.config.cat_features]
        self.y = self.df[target]

        # Create preprocessing steps for numeric and categorical data
        numeric_transformer = Pipeline(
            steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
        )

        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )

        # Combine preprocessing steps
        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, self.config.num_features),
                ("cat", categorical_transformer, self.config.cat_features),
            ]
        )

    @log_execution_time("Split data in training and testing sets.")
    def split_data(self, test_size: float = 0.2, random_state: float = 42):
        """
        Split the data into training and testing sets.

        Args:
            test_size (float, optional): Proportion of the dataset to
                include in the test split. Defaults to 0.2.
            random_state (int, optional): Random state for
                reproducibility. Defaults to 42.

        Returns:
            tuple: (X_train, X_test, y_train, y_test) - Split feature and target datasets.
        """
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state
        )

        logger.debug(f"Training set shape: {X_train.shape}, Test set shape: {X_test.shape}")

        return X_train, X_test, y_train, y_test

    @log_execution_time("Save data to catalog.")
    def save_to_catalog(self, train_set: pd.DataFrame, test_set: pd.DataFrame, spark: SparkSession):
        """Save the train and test sets into Databricks tables."""

        train_set_with_timestamp = spark.createDataFrame(train_set).withColumn(
            "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC")
        )

        test_set_with_timestamp = spark.createDataFrame(test_set).withColumn(
            "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC")
        )

        train_set_with_timestamp.write.mode("append").saveAsTable(
            f"{self.config.catalog_name}.{self.config.schema_name}.train_set"
        )

        test_set_with_timestamp.write.mode("append").saveAsTable(
            f"{self.config.catalog_name}.{self.config.schema_name}.test_set"
        )

        spark.sql(
            f"ALTER TABLE {self.config.catalog_name}.{self.config.schema_name}.train_set "
            "SET TBLPROPERTIES (delta.enableChangeDataFeed = true);"
        )

        spark.sql(
            f"ALTER TABLE {self.config.catalog_name}.{self.config.schema_name}.test_set "
            "SET TBLPROPERTIES (delta.enableChangeDataFeed = true);"
        )

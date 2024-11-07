"""This module contains a class for processing video game sales data."""

import pandas as pd
from pandas import DataFrame as PandasDataFrame
from pyspark.sql import SparkSession
from pyspark.sql.dataframe import DataFrame as SparkDataFrame
from pyspark.sql.functions import current_timestamp, to_utc_timestamp
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import config as default_config
from src.utilies.decorators import log_execution_time
from src.video_game_sales.config import ProjectConfig


class DataProcessor:
    """
    A class for processing video game sales data.
    This class handles loading, preprocessing, and splitting the data
        for machine learning tasks.
    """

    @log_execution_time("Initialize DataProcessor.")
    def __init__(self, config: ProjectConfig = None, pandas_df: PandasDataFrame = None) -> None:
        """
        Initialize the DataProcessor.

        Args:
            filepath (str): Path to the CSV file containing the data.
            config (dict): Configuration dictionary containing feature
                and target information.
        """
        self.df = pandas_df
        self.config = config or default_config
        self.X = None
        self.y = None
        self.preprocessor = None

    @log_execution_time("Load data.")
    def load_data(self, filepath: str) -> PandasDataFrame:
        """
        Load data from a CSV file.

        Args:
            filepath (str): Path to the CSV file.

        Returns:
            pandas.DataFrame: Loaded data.
        """
        self.df = pd.read_csv(filepath)
        return self.df

    @log_execution_time("Data preprocessing.")
    def preprocess_data(self) -> None:
        """
        Preprocess the data by aggregating sales columns, removing rows with
        missing target, separating features and target, and creating
        preprocessing pipelines.
        """

        # Remove rows with missing target
        target = self.config.target
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        self.df = self.df.dropna()

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
            tuple: (train_set, test_set) - Split feature and target datasets.
        """

        train_set, test_set = train_test_split(self.df, test_size=test_size, random_state=random_state)
        return train_set, test_set

    @log_execution_time("Save data to catalog.")
    def save_to_catalog(self, train_set: PandasDataFrame, test_set: PandasDataFrame, spark: SparkSession) -> None:
        """Save the train and test sets into Databricks tables."""

        train_set_with_timestamp = spark.createDataFrame(train_set).withColumn(
            "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC")
        )

        test_set_with_timestamp = spark.createDataFrame(test_set).withColumn(
            "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC")
        )

        train_set_with_timestamp.write.mode("overwrite").saveAsTable(
            f"{self.config.catalog_name}.{self.config.schema_name}.train_set"
        )

        test_set_with_timestamp.write.mode("overwrite").saveAsTable(
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

    @log_execution_time("Load data from catalog - SPARK.")
    def load_from_catalog_spark(self, spark: SparkSession) -> tuple[SparkDataFrame, SparkDataFrame]:
        """Load the train and test sets from Databricks tables.
        Args:
            spark (SparkSession): Spark session object.
        Returns:
            tuple: (train_set, test_set) - Train and test sets as Spark DataFrames.
        """

        train_set = spark.table(f"{self.config.catalog_name}.{self.config.schema_name}.train_set")
        test_set = spark.table(f"{self.config.catalog_name}.{self.config.schema_name}.test_set")
        return train_set, test_set

    @log_execution_time("Load data from catalog - PANDAS.")
    def load_from_catalog_pandas(self, spark: SparkSession) -> tuple[PandasDataFrame, PandasDataFrame]:
        """Load the train and test sets from Databricks tables as Pandas DataFrames.
        Args:
            spark (SparkSession): Spark session object.
        Returns:
            tuple: (train_set, test_set) - Train and test sets as Pandas DataFrames.
        """

        train_set, test_set = self.load_from_catalog_spark(spark)
        return train_set.toPandas(), test_set.toPandas()

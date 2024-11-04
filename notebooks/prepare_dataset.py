"""Prepare the dataset for training and testing."""

# Databricks notebook source
from pyspark.sql import SparkSession

from src.video_game_sales import DataProcessor
from src.video_game_sales.config import ProjectConfig

spark = SparkSession.builder.getOrCreate()

# COMMAND ----------

config = ProjectConfig.from_yaml()

# COMMAND ----------
# Load the house prices dataset
url = config.data.url
df = spark.read.csv(config.data_full_path, header=True, inferSchema=True).toPandas()

# COMMAND ----------
data_processor = DataProcessor(pandas_df=df, config=config)
data_processor.preprocess()
train_set, test_set = data_processor.split_data()
data_processor.save_to_catalog(train_set=train_set, test_set=test_set, spark=spark)

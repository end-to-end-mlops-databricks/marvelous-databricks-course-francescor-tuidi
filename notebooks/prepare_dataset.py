# Databricks notebook source
from pyspark.sql import SparkSession

from src import config
from src.video_game_sales.data_processor import DataProcessor

# COMMAND ----------

spark = SparkSession.builder.getOrCreate()

# COMMAND ----------

url = "/" + config.data_full_path
df = spark.read.csv(url, header=True, inferSchema=True).toPandas()

# COMMAND ----------

data_processor = DataProcessor(pandas_df=df, config=config)
data_processor.preprocess_data()
train_set, test_set = data_processor.split_data()

# COMMAND ----------

data_processor.save_to_catalog(train_set=train_set, test_set=test_set, spark=spark)

import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import current_timestamp, to_utc_timestamp

from src import config
from src.utils.synthetic_data import create_synthetic_data

spark = SparkSession.builder.getOrCreate()

catalog_name = config.catalog_name
schema_name = config.schema_name

# Load train and test sets
train_set = spark.table(f"{catalog_name}.{schema_name}.train_set").toPandas()
test_set = spark.table(f"{catalog_name}.{schema_name}.test_set").toPandas()
combined_set = pd.concat([train_set, test_set], ignore_index=True)
combined_set = combined_set.drop("update_timestamp_utc", axis=1)


# Create synthetic data
synthetic_df = create_synthetic_data(combined_set)

existing_schema = spark.table(f"{catalog_name}.{schema_name}.source_data").schema

synthetic_spark_df = spark.createDataFrame(synthetic_df, schema=existing_schema)

train_set_with_timestamp = synthetic_spark_df.withColumn(
    "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC")
)

# Append synthetic data as new data to source_data table
train_set_with_timestamp.write.mode("overwrite").saveAsTable(f"{catalog_name}.{schema_name}.source_data")

# Check
syntehtic_data_spark = spark.table(f"{catalog_name}.{schema_name}.source_data")
syntehtic_data_spark.show(5)
syntehtic_data_spark.count()

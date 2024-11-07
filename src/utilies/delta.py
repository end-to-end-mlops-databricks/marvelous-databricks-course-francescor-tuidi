"""This module contains utility functions for Delta tables."""

from delta.tables import DeltaTable
from pyspark.sql import SparkSession

from src import logger


def get_latest_delta_version(table_path: str, spark: SparkSession) -> int:
    """Get the latest version of a Delta table.

    Args:
        table_path (str): The path of the Delta table.
        spark (SparkSession): The Spark session.

    Returns:
        int: The latest version of the Delta table.
    """
    delta_table = DeltaTable.forPath(spark, table_path)
    delta_version = delta_table.history().select("version").collect()
    logger.info(f"Latest Delta version: {delta_version[-1]['version']}")
    return delta_version[-1]["version"]

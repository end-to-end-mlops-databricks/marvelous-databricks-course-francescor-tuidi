# Databricks notebook source
import json
import os

import mlflow

from notebooks import config
from src import logger

# COMMAND ----------

mlflow.set_tracking_uri("databricks")

experiments_path = os.path.join("/Shared", f"{config.catalog_name}-{config.schema_name}")
mlflow.set_experiment(experiment_name=experiments_path)
mlflow.set_experiment_tags({"repository_name": config.schema_name})

# COMMAND ----------

experiments = mlflow.search_experiments(filter_string=f"tags.repository_name='{config.schema_name}'")
logger.info(experiments)

# COMMAND ----------

with open("mlflow_experiment.json", "w") as json_file:
    json.dump(experiments[0].__dict__, json_file, indent=4)

# COMMAND ----------

with mlflow.start_run(
    run_name="demo-run",
    tags={"git_sha": "ffa63b430205ff7", "branch": "week2"},
    description="demo run",
) as run:
    mlflow.log_params({"type": "demo"})
    mlflow.log_metrics({"metric1": 1.0, "metric2": 2.0})

# COMMAND ----------

run_id = mlflow.search_runs(
    experiment_names=[experiments_path],
    filter_string="tags.git_sha='ffa63b430205ff7'",
).run_id[0]
run_info = mlflow.get_run(run_id=f"{run_id}").to_dictionary()
logger.info(run_info)

# COMMAND ----------

with open("run_info.json", "w") as json_file:
    json.dump(run_info, json_file, indent=4)

# COMMAND ----------

logger.info(run_info["data"]["metrics"])

# COMMAND ----------

logger.info(run_info["data"]["params"])

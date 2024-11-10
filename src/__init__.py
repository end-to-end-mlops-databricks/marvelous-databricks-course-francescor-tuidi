"""Init file."""

import logging as logger

import yaml

logger.basicConfig(level=logger.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

with open("project_config.yml", "r") as file:
    config = yaml.safe_load(file)

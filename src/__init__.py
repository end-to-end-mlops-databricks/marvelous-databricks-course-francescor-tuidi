"""Init file."""

import logging as logger

from src.video_game_sales.config import ProjectConfig

PROJECT_CONFIG_PATH = "project_config.yml"

try:
    config = ProjectConfig.from_yaml(config_path=PROJECT_CONFIG_PATH)
except FileNotFoundError:
    # Notebooks on databricks can not find the config file with absolute paths
    config = ProjectConfig.from_yaml(config_path=f"../{PROJECT_CONFIG_PATH}")

logger.basicConfig(
    level=logger.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

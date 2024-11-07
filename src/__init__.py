"""Init file."""

import logging as logger

from src.video_game_sales.config import ProjectConfig

config = ProjectConfig.from_yaml(config_path="project_config.yml")

logger.basicConfig(level=logger.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

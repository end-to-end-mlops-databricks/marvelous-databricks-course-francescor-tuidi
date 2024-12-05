"""Configuration module."""

import os
from typing import Any, Dict, List

import yaml
from pydantic import BaseModel


class ProjectConfig(BaseModel):
    volumes_root: str
    catalog_name: str
    schema_name: str
    data_path: str
    data_full_path: str  # Built from the other fields.
    target: str
    parameters: Dict[str, Any]
    num_features: List[str]
    cat_features: List[str]
    ab_test: Dict[str, Any]

    @classmethod
    def from_yaml(cls, config_path: str = "project_config.yml"):
        """Load configuration from a YAML file.
        Args:
            config_path (str): The path to the configuration file.
        Returns:
            ProjectConfig: The configuration object.
        """
        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)
            config_dict["data_full_path"] = os.path.join(
                config_dict["volumes_root"],
                config_dict["catalog_name"],
                config_dict["schema_name"],
                config_dict["data_path"],
            )
        return cls(**config_dict)

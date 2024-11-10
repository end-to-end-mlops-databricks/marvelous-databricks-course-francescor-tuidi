"""Util module for interacting with Git."""

import git

from src import logger


def get_git_info() -> tuple[str, str]:
    """Gets the Git SHA and the current branch.

    Returns:
        tuple[str, str]: A tuple containing the Git SHA and the current branch.
    """
    try:
        repo = git.Repo(search_parent_directories=True)
        git_sha = repo.head.object.hexsha
        current_branch = repo.active_branch.name
    except git.exc.InvalidGitRepositoryError:
        # In case of a Databricks notebook, the git repo is not available
        git_sha = "latest"
        current_branch = "latest"
    logger.info(f"Git SHA: {git_sha}")
    logger.info(f"Branch: {current_branch}")
    return git_sha, current_branch

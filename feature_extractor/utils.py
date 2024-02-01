
# fun: get_pose_model_dir
from pathlib import Path
import sys
from loguru import logger

# fun: time logger
from typing import Optional
import time

import numpy as np

def get_project_dir_path() -> Path:
    for parent in Path(__file__).parents:
        if (parent / "pyproject.toml").exists():
            return parent


def get_pose_model_dir() -> Path:
    # Recursively ascend the parent directories of this file's path looking for
    # the .venv folder.
    for parent in Path(__file__).parents:
        if (parent / "pyproject.toml").exists():
            return parent / "models"

    # If the .venv folder could not be found, just use the current working
    # directory.
    raise Exception("Do not find pyproject.toml file. Please Init the Project first")


logger.remove(0)
logger.add(
    sys.stdout,
    colorize=True,
    format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level}</level> | <cyan>{file}:{line}</cyan> - <level>{message}</level>",
)


def time_logger(pre: float | None = None) -> float | None:
    cur = time.time()
    if pre is None:
        logger.info("Computational Analysis Starts")
        return cur
    else:
        logger.info(f"It took {cur-pre} seconds")
        return None
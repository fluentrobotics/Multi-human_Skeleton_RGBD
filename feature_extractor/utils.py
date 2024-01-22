
# fun: get_pose_model_dir
from pathlib import Path
import sys
from loguru import logger

# fun: plot_3d_sactters_with_mask
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def get_pose_model_dir() -> Path:
    # Recursively ascend the parent directories of this file's path looking for
    # the .venv folder.
    for parent in Path(__file__).parents:
        if (parent / "pyproject.toml").exists():
            return parent / "models"

    # If the .venv folder could not be found, just use the current working
    # directory.
    return Path("../models")


logger.remove(0)
logger.add(
    sys.stdout,
    colorize=True,
    format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level}</level> | <cyan>{file}:{line}</cyan> - <level>{message}</level>",
)


def plot_3d_sactters_with_mask(plot_mat: np.ndarray, mask_mat: np.ndarray, sc):
    """
    @plot_mat: [H, K, 3]
    @mask_mat: [H, K]
    @fig_config: list[fig, ax, sc]
    return fig_config
    """

    scatter_mat = plot_mat.reshape(-1,3)        # [HK,3]
    scatter_mask = mask_mat.reshape(-1)         # [HK]

    scatter_mat = scatter_mat[scatter_mask]     # [HK-,3]
    
    if fig_config is None:
        plt.ion()
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        sc = ax.scatter(scatter_mat[:, 0], scatter_mat[:, 1], scatter_mat[:, 2])
        fig.show()

    else:
        fig = fig_config[0]
        ax = fig_config[1]
        sc = fig_config[2]

        sc._offsets3d = (scatter_mat[:,0], scatter_mat[:,1], scatter_mat[:,2])
        # plt.draw()
    
    fig_config = [fig, ax, sc]
    return fig_config

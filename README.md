# Multi-human_Skeleton_RGB-D
Efficient multi-human 3D skeleton tracking based on RealSenese RGBD

DEMO
![image](/doc/test_Outliers_True_fig154.png)

## Dependency Version

Python 3.10

numpy, ultralytics, torch, opencv-python

Details in [pyproject.toml](/pyproject.toml)

[Poetry](https://python-poetry.org/) is recommended to initialize this repository, where [pyproject.toml](/pyproject.toml) and [poetry.lock](/poetry.lock) are provided.

## Get started
1. Set global configuration [config.py](/feature_extractor/config.py)
You can modify YOLO-POSE model `POSE_MODEL`, filters `USE_KALMAN` `MINIMAL_FILTER` `OUTLIER_FILTER`, output filename `TASK_NAME`, save mode `SAVE_YOLO_IMG`.

2. Run `feature_extractor/skeleton_extractor_node.py` with activated ROS TOPICS. NOTE: make sure the image types in configuaration match your ROS TOPICS(Imgae vs Compressed Image)

3. The results will be saved in `data/piclke` as `.pkl` with `TASK_NAME`.

4. If you want to generate a demo, run `plot/plot_pickle.py` and you will get Matplotlib figures generated in `data/figure`. Then run `plot/creat_video_from_img.py` and get video demo in `data/video`. NOTE: The opencv and matplotlib might conflict because of PyQt5 and cause dump conflicts, try to avoid `import` them in the same PID.
# 3D Multi-human Skeleton with RGB-D

This repository aims to develop an Efficient 3D Multi-human Pose Estimation. Inference efficiency is about 20 Hz with Nvidia A2000. We starts from the RGB-D ROS image/compressed image to the 3D keypoint representation in the camera coordinate.

Feel free to choose any filter you want to compare the their performace.

## DEMO

<img src="/doc/test_Outliers_True_fig154.png" width="500"/>

GIF

<img src="/doc/Multi_Human2_ani.gif" width="500"/>

<!-- ![image](/doc/test_Outliers_True_fig154.png =250x) -->

<!-- ![image](/doc/test_Outliers_True_ani.gif) -->


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

4. If you want to generate a demo, run `plot/plot_pickle.py` and you will get Matplotlib figures generated in `data/figure`. 

5. Then run `plot/creat_video_from_img.py` and get video demo in `data/video`. Before generate the video, you can select desired frame interval in `config`. 
NOTE: The opencv and matplotlib might conflict because of PyQt5 and cause dump conflicts, try to avoid `import` them in the same PID.


## License
ultralytics YOLO v8 requires all the following repository with [**GNU AGPLv3**](/LICENSE).


## What's Next
* ROS2 dev, Real-time deployment on robot and calibration with multi-view pseudo ground true.
* Descriptor and Feature
* Action prediction based motion planning.
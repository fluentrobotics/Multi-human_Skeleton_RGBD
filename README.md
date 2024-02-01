# Multi-human_Skeleton_RGB-D
Efficient multi-human 3D skeleton tracking based on RealSenese RGBD

DEMO
![image](/doc/test_Outliers_True_fig154.png)

## Getting Started

### dependency version

Python 3.10

numpy, ultralytics, torch, opencv-python

Details in [pyproject.toml](/pyproject.toml)


### Get started
Set Parameters(TASK NAME and which filter) in global config file [config.py](/feature_extractor/config.py)

run 'feature_extractor/skeleton_extractor_node.py'

run ros system or rosbag

get '.pkl' results 

run 'plot/plot_pickle.py' and get figure demo in 'data/figure'

run 'plot/creat_video_from_img.py' and get video demo in 'data/video'

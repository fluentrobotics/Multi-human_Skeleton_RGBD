import cv2
import matplotlib.pyplot as plt
from feature_extractor.config import *

import os
if not os.path.exists(DATA_DIR_PATH / "pickle"):
    os.mkdir(DATA_DIR_PATH / "pickle")

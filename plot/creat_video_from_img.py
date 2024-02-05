import glob
import re

import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from feature_extractor.config import *

fig_path = DATA_DIR_PATH / "figure" / TEST_NAME
fig_path = fig_path.absolute().as_posix() + "_fig"

video_path = DATA_DIR_PATH / "video" / TEST_NAME
video_path = video_path.absolute().as_posix() + "_video.mp4"

img_array = []
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

pattern = f"{fig_path}*.png"
files = glob.glob(pattern)

def extract_number(filename):
    match = re.search(r"(\d+)\.png$", filename)
    return int(match.group(1)) if match else 0

files_sorted = sorted(files, key=extract_number)

for filename in tqdm(files_sorted,
               ncols=80,
               colour="red",
               ):
    
    img = cv2.imread(filename)
    if img is None:
        continue
    height, width, layers = img.shape
    size = (width,height)
    img_array.append(img)
    # print(id)
    
out = cv2.VideoWriter(video_path, fourcc, PUB_FREQ, size)
 
for i in range(len(img_array)):
    out.write(img_array[i])

cv2.destroyAllWindows()
out.release()
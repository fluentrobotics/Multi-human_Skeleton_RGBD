import cv2
import numpy as np
from pathlib import Path


img_array = []
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
count_max = 156
for id in range(count_max+1):
    filename = f"/home/xmo/socialnav_xmo/feature_extractor/img/minimal_false/minimal_false{id}.png"
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width,height)
    img_array.append(img)

out = cv2.VideoWriter('/home/xmo/socialnav_xmo/feature_extractor/img/minimal_false.avi',fourcc,10, size)
 
for i in range(len(img_array)):
    out.write(img_array[i])

cv2.destroyAllWindows()
out.release()
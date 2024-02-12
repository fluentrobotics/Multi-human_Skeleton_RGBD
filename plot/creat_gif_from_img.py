import glob
import re

import cv2
import imageio
import numpy as np
from pathlib import Path
from tqdm import tqdm
from feature_extractor.config import *

fig_path = DATA_DIR_PATH / "figure" / TEST_NAME
fig_path = fig_path.absolute().as_posix() + "_fig"

gif_path = DATA_DIR_PATH / "video" / TEST_NAME
gif_path = gif_path.absolute().as_posix() + "_ani.gif"

img_array = []

pattern = f"{fig_path}*.png"
files = glob.glob(pattern)

def extract_number(filename):
    match = re.search(r"(\d+)\.png$", filename)
    return int(match.group(1)) if match else 0

files_sorted = sorted(files, key=extract_number)

step = 0

with imageio.get_writer(gif_path, mode="I", fps=PUB_FREQ, loop=0) as writer:
    for filename in tqdm(files_sorted,
                ncols=80,
                colour="red",
                ):
        
        if (step >= VIDEO_START and step <= VIDEO_END) and SELECTED_VIDEO_REGION:
            image = imageio.imread(filename)
            writer.append_data(image)
        
        step += 1

print(f"Done! GIF path: {gif_path}")
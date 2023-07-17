# This is a script for trimming video files to 10secs

from moviepy.editor import *
import os
from tqdm import tqdm

# Video input path
real_path = 'Celeb-synthesis'

# Features output path
feature_path_r = 'Features/fake/'
feature_map_count = 0

print("Beginning trimming----------------------")
# Iterate over the files in the directory
for filename in tqdm(os.listdir(real_path)):
    f = os.path.join(real_path, filename)
    feature_map_count += 1
    map_name = '.mp4'
    
    # Use the VideoFileClip class from moviepy library
    clip = VideoFileClip(f)
    
    # Trim the the video to 10 secs
    subclip1 = clip.subclip((0,00), (0,10))
    subclip1.write_videofile(feature_path_r+str(feature_map_count)+map_name)
    
print('End Trimming----------------------------')
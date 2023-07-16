from moviepy.editor import *
import os
from tqdm import tqdm

real_path = 'Celeb-synthesis'
feature_path_r = 'Features/fake/'
feature_map_count = 0

print("Beginning trimming----------------------")
for filename in tqdm(os.listdir(real_path)):
    f = os.path.join(real_path, filename)
    feature_map_count += 1
    map_name = '.mp4'
    
    clip = VideoFileClip(f)
    subclip1 = clip.subclip((0,00), (0,10))
    subclip1.write_videofile(feature_path_r+str(feature_map_count)+map_name)
    
print('End Trimming----------------------------')
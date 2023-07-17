import os
from tqdm import tqdm
import pandas as pd

# Feature map paths
map_path_fake = 'Feature_Maps/Fake/'
map_path_real = 'Feature_Maps/Real/'

# Dataset path
dataset_path = 'datasetv1/dsv1.csv'

# Feature list
feature_list = []

# Iterate over fake feature maps and append to feature list
for filename in tqdm(os.listdir(map_path_fake)):
    map_file = os.path.join(map_path_fake, filename)
    
    df = pd.read_csv(map_file)
    feature_list.append(df)
    
# Iterate over fake feature maps and append to feature list
for filename in tqdm(os.listdir(map_path_real)):
    map_file = os.path.join(map_path_real, filename)
    
    df = pd.read_csv(map_file)
    feature_list.append(df)
    
# Feature map dataframe
feature_frame = pd.concat(feature_list, axis=0, ignore_index=True)
feature_frame.to_csv(dataset_path)
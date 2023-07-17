# This is a script for extracting and saving the action unit intensity columns from the feature maps

import os
from tqdm import tqdm
import pandas as pd

# Features input path
feature_path = 'Features/fake/features'

# Feature map output path
out_path = 'Feature_Maps/Fake/'
feature_count = 0

print("Beginning extraction----------------------")
for filename in tqdm(os.listdir(feature_path)):
    f = os.path.join(feature_path, filename)
    feature_count += 1
    
    df = pd.read_csv(f)
    
    # Get the action unit intensity columns from the dataframe
    df = df.loc[:, ' AU01_r':' AU45_r']
    # Strip whitespace
    df.columns = df.columns.str.replace(' ', '')
    # Set label fake or real
    df['label'] = 'fake'
    
    save_path = out_path + str(feature_count) + '_fake.csv'
    df.to_csv(save_path)
    
print("Extraction complete----------------------")
    # print(df.head())
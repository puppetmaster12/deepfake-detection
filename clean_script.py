import os
from tqdm import tqdm
import pandas as pd

feature_path = 'Features/fake/features'
out_path = 'Feature_Maps/Fake/'
feature_count = 0

print("Beginning extraction----------------------")
for filename in tqdm(os.listdir(feature_path)):
    f = os.path.join(feature_path, filename)
    feature_count += 1
    
    df = pd.read_csv(f)
    df = df.loc[:, ' AU01_r':' AU45_r']
    df.columns = df.columns.str.replace(' ', '')
    df['label'] = 'fake'
    
    save_path = out_path + str(feature_count) + '_fake.csv'
    df.to_csv(save_path)
    
print("Extraction complete----------------------")
    # print(df.head())
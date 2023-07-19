import os
import pandas as pd
from tqdm import tqdm

fake_maps_path = 'Feature_Maps/Fake/'
real_maps_path = 'Feature_Maps/Real/'
dataset_path = 'datasetv1/dsv1.csv'

# for filename in tqdm(os.listdir(dataset_path)):
#     file_path = os.path.join(dataset_path, filename)
#     df = pd.read_csv(file_path)
#     print(df.head())

df = pd.read_csv(dataset_path)
df.loc[df['label'] == 'fake', 'label'] = 1
df.loc[df['label'] == 'real', 'label'] = 0

df.to_csv('datasetv1/dsv1_1.csv')
import os
import pandas as pd

data_path = 'C:\\Users\\pavit\\Documents\\Pavin\\MDA\\MDA692\\Project\\CelebDF\\datasetv1\\features'
count = 0
for filename in os.listdir(data_path):
    
    file_path = os.path.join(data_path, filename)
    print(filename)
    if filename.endswith('.csv'):
        df = pd.read_csv(file_path)
        if(len(df) > 300):
            count += 1
            print(filename, df.shape)

print(count)
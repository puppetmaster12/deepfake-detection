import os
import pandas as pd
from tqdm import tqdm

class VideoPreprocessor:
    def __init__(self, feature_path, output_dir, label):
        self.feature_path = feature_path
        self.output_dir = output_dir
        self.label = label
    
    # Function for preprocessing OpenFace feature maps
    def preprocess(self):
        feature_count = 0
        for filename in tqdm(os.listdir(self.feature_path)):
            f = os.path.join(self.feature_path, filename)
            feature_count += 1
            
            df = pd.read_csv(f)
            
            # Get the action unit intensity columns from the dataframe
            df = df.loc[:, ' AU01_r':' AU45_r']
            
            # Strip whitespace
            df.columns = df.columns.str.replace(' ', '')
            
            # Fill row with 0 if less than 300 frames
            frame_count = 300 - len(df)
            if(frame_count > 0):
                for i in range(frame_count):
                    df.loc[len(df) + i] = 0
            
            
            
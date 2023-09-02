import os
import pandas as pd
from tqdm import tqdm
from moviepy.editor import VideoFileClip
import argparse

class VideoPreprocessor:
    def __init__(self, feature_path, output_dir):
        self.feature_path = feature_path
        self.output_dir = output_dir
    
    # Function for preprocessing OpenFace feature maps
    def preprocess_csv(self):
        for filename in tqdm(os.listdir(self.feature_path)):
            f = os.path.join(self.feature_path, filename)
            
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
            
            df = df.loc[:,~df.columns.str.startswith('Unnamed')]
            
            output_name = os.path.join(self.output_dir, filename)
            df.to_csv(output_name)
            
    def trim_video(self, videos_path, output_path):
        for filename in os.listdir(videos_path):
            if filename.endswith(".mp4"):
                f = os.path.join(videos_path, filename)
                clip = VideoFileClip(f)
                
                subclip1 = clip.subclip((0,00), (0,10))
                subclip1.write_videofile(os.path.join(output_path, filename))
                
def main():
    parser = argparse.ArgumentParser(description="Preprocessing class for both csv feature maps and trimming function for videos")
    parser.add_argument("--trim", required=False, help="Used when trimming videos")
    parser.add_argument("--input_dir", required=True, help="Path to the csv or video files for preprocessing")
    parser.add_argument("--output_dir", required=True, help="Path to save the preprocessed files")
    args = parser.parse_args()
    
    preprocessor = VideoPreprocessor(args.input_dir, args.output_dir)
    
    if args.trim == "true":
        preprocessor.trim_video(args.input_dir, args.output_dir)
    else:
        preprocessor.preprocess_csv(args.input_dir, args.output_dir)    
    
    
    
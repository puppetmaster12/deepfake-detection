import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet18
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import LabelEncoder
import argparse
from tqdm import tqdm

# Define a custom dataset class to load and preprocess videos
class VideoDataset(Dataset):
    def __init__(self, video_dir, transform=None):
        self.video_dir = video_dir
        self.video_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
        self.transform = transform

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        video_filename = os.path.join(self.video_dir, self.video_files[idx])
        cap = cv2.VideoCapture(video_filename)
        frames = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if self.transform:
                frame = self.transform(frame)
            frames.append(frame)
        
        cap.release()

        return torch.stack(frames)

# Define the feature extractor using a pre-trained ResNet-18 model
class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        resnet = resnet18(pretrained=True)
        self.features = nn.Sequential(*list(resnet.children())[:-1])  # Remove the final classification layer

    def forward(self, x):
        return self.features(x)

# Function to extract features from videos
def extract_features_from_videos(input_video_dir, output_feature_dir):
    # Create the output directory if it doesn't exist
    os.makedirs(output_feature_dir, exist_ok=True)

    # Load the custom ResNet-based feature extractor
    feature_extractor = FeatureExtractor()
    feature_extractor.eval()  # Set to evaluation mode (no gradient computation)

    # Define transformations for frames (resize and normalize)
    transform = transforms.Compose([transforms.ToPILImage(),  # Convert to PIL Image
                                    transforms.Resize((224, 224)),  # Resize to the input size of ResNet
                                    transforms.ToTensor(),  # Convert to Tensor
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                         std=[0.229, 0.224, 0.225])])

    # Create a custom dataset for video frames
    video_dataset = VideoDataset(input_video_dir, transform=transform)

    # Loop through each video and extract features
    for i, video_batch in tqdm(enumerate(video_dataset)):
        # Extract features for this video batch
        with torch.no_grad():  # Disable gradient computation for inference
            features = feature_extractor(video_batch)
        
        # Reshape features to match the frame count
        num_frames = features.shape[0]
        features = features.view(num_frames, -1).numpy()
        
        # Load the labels and encode them (you should load labels from your dataset)
        labels = []  # You should load the labels from your dataset
        # label_encoder = LabelEncoder()
        # labels_encoded = label_encoder.fit_transform(labels)
        
        # Create a DataFrame to store features and labels
        df = pd.DataFrame(features)
        
        # Add the encoded labels to the DataFrame
        # df['Label'] = labels_encoded
        
        # Get the video filename and create the output path
        video_filename = video_dataset.video_files[i]
        output_csv_path = os.path.join(output_feature_dir, os.path.splitext(video_filename)[0] + '.csv')
        
        # Save the DataFrame to a separate CSV file for each video
        df.to_csv(output_csv_path, index=False)
        print(f"Saved features for {video_filename} to {output_csv_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Video Feature Extraction")
    parser.add_argument("--input_dir", required=True, help="Input video directory")
    parser.add_argument("--output_dir", required=True, help="Output feature directory")
    args = parser.parse_args()

    input_video_dir = args.input_dir
    output_feature_dir = args.output_dir

    extract_features_from_videos(input_video_dir, output_feature_dir)

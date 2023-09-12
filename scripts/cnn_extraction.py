import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet18
from sklearn.preprocessing import LabelEncoder
import argparse

# Define a custom dataset class to load and preprocess images
class ImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg') or f.endswith('.png')]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_filename = os.path.join(self.image_dir, self.image_files[idx])
        image = cv2.imread(image_filename)
        
        if self.transform:
            image = self.transform(image)
        
        return image

# Define the feature extractor using a pre-trained ResNet-18 model
class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        resnet = resnet18(pretrained=True)
        self.features = nn.Sequential(*list(resnet.children())[:-1])  # Remove the final classification layer

    def forward(self, x):
        return self.features(x)

# Function to extract features from images
def extract_features_from_images(input_image_dir, output_feature_dir):
    # Create the output directory if it doesn't exist
    os.makedirs(output_feature_dir, exist_ok=True)

    # Load the custom ResNet-based feature extractor
    feature_extractor = FeatureExtractor()
    feature_extractor.eval()  # Set to evaluation mode (no gradient computation)

    # Define transformations for images (resize and normalize)
    transform = transforms.Compose([transforms.ToPILImage(),  # Convert to PIL Image
                                    transforms.Resize((224, 224)),  # Resize to the input size of ResNet
                                    transforms.ToTensor(),  # Convert to Tensor
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                         std=[0.229, 0.224, 0.225])])

    # Loop through each subdirectory (representing each video)
    for video_folder in os.listdir(input_image_dir):
        video_folder_path = os.path.join(input_image_dir, video_folder)
        
        if not os.path.isdir(video_folder_path):
            continue  # Skip non-directory files
        
        # Create a custom dataset for images in this subdirectory
        image_dataset = ImageDataset(video_folder_path, transform=transform)
        
        # Extract features from images
        features_list = []
        
        for image_batch in image_dataset:
            # Extract features for this image batch
            with torch.no_grad():  # Disable gradient computation for inference
                features = feature_extractor(image_batch.unsqueeze(0))
            features_list.append(features)
        
        # Stack features for all images in this subdirectory
        features = torch.cat(features_list)
        
        # Reshape features to match the number of images
        num_images = features.shape[0]
        features = features.view(num_images, -1).numpy()
        
        # Load the labels and encode them (you should load labels from your dataset)
        labels = []  # You should load the labels from your dataset
        label_encoder = LabelEncoder()
        labels_encoded = label_encoder.fit_transform(labels)
        
        # Create a DataFrame to store features and labels
        df = pd.DataFrame(features)
        
        # Add the encoded labels to the DataFrame
        # df['Label'] = labels_encoded
        
        # Create the output CSV file path
        output_csv_path = os.path.join(output_feature_dir, f'{video_folder}.csv')
        
        # Save the DataFrame to a separate CSV file for each video
        df.to_csv(output_csv_path, index=False)
        print(f"Saved features for {video_folder} to {output_csv_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image Feature Extraction")
    parser.add_argument("--input_dir", required=True, help="Input image directory containing subdirectories for each video")
    parser.add_argument("--output_dir", required=True, help="Output feature directory")
    args = parser.parse_args()

    input_image_dir = args.input_dir
    output_feature_dir = args.output_dir

    extract_features_from_images(input_image_dir, output_feature_dir)

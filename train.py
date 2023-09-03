import numpy as np
import os
from tqdm import tqdm
import pandas as pd
import json
from scripts.preprocessor import VideoPreprocessor
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# Function to load csv files to numpy array
def load_feature(feature_path):
    df = pd.read_csv(feature_path, index_col=0, nrows=300)
        
    frame_features = df.to_numpy()
    
    return frame_features

def load_labels(label_path):
    with open(label_path, "r") as json_file:
        labels = json.load(json_file)
    
    for key in labels:
        labels[key] = int(labels[key])
        
    return labels

# Dataset class with a dataloader
class VideoDataset(Dataset):
    def __init__(self, video_data, labels, sequence_length):
        self.video_data = video_data
        self.labels = labels
        self.sequence_length = sequence_length
        
    def __len__(self):
        return len(self.video_data)
    
    # Function to get the frames and label of a video by index
    def __getitem__(self, index):
        frames = self.video_data[index]
        label = self.labels[index]
        
        return frames, label
    
# Main LSTM model class
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.bn = nn.BatchNorm1d(hidden_size)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )
        
    # One cycle of forward propogation
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.bn(out)
        out = self.fc(out)
        return out

video_data = []
labels = []
dataset_path = 'datasetv1/features/final'
labels_path = 'datasetv1/labels/labels.json'

# Load labels
labels_all = load_labels(labels_path)

# Iterate over csv files and load into list as numpy array
print("Loading Dataset ==================================")
for filename in tqdm(os.listdir(dataset_path)):
    if filename.endswith('.csv'):
        file_path = os.path.join(dataset_path, filename)
        video_id = filename.split(".")[0]
        labels.append(labels_all[video_id])
        
        frame_features = load_feature(file_path)
        video_data.append(frame_features)
print("Dataset loaded ===================================")
print(len(labels))
# Put all video sequences / frames into one dataset
sequence_length = 300 # 300 frames per video
dataset = VideoDataset(video_data, labels, sequence_length)

# Dataset split
train_size = int(0.7 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

# Pytorch dataloader for training and validation data
batch_size = 128
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# Model Configurations
input_size = len(video_data[0][0]) # Total number of features
hidden_size = 32
num_layers = 8 # TODO: tune
output_size = 1 # TODO: tune

# Model initialization
lstm_model = LSTMModel(input_size, hidden_size, num_layers, output_size)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
lstm_model.to(device)

# Loss function and Optimizer function
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(lstm_model.parameters(), lr=0.0001) # TODO: tune

# Start Training
num_of_epochs = 930 # TODO: tune

for epoch in range(num_of_epochs):
    lstm_model.train()
    for frames, labels in train_loader:
        
        frames = frames.to(torch.float32).to(device)
        labels = labels.to(torch.float32).unsqueeze(1).to(device)
        
        # Forward propogate
        outputs = lstm_model(frames)
        loss = criterion(outputs, labels)
        
        # Backward propogate and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Print out the training performance stats
        print(f'Epoch [{epoch + 1}/{num_of_epochs}], Loss: {loss.item():.4f}') # TODO: add more metrics
    
    # Model validation
    lstm_model.eval()
    y_true = [] # Actual label
    y_pred = [] # Predicted label
    
    with torch.no_grad():
        total_correct = 0
        total_samples = 0
        for frames, labels in val_loader:
            frames = frames.to(torch.float32).to(device)
            labels = labels.to(torch.float32).to(device)
            
            outputs = lstm_model(frames)
            predicted_labels = (outputs > 0.5).float()
            
            total_correct += (predicted_labels == labels).sum().item()
            total_samples += labels.size(0)
            
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted_labels.cpu().numpy())
        
        val_accuracy = total_correct / total_samples
        print(f'Validation Accuracy: {val_accuracy:.4f}')
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_pred)
    conf_matrix = confusion_matrix(y_true, y_pred)
    
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')
    print(f'ROC AUC Score: {roc_auc:.4f}')
    print('Confusion Matrix:')
    print(conf_matrix)
    
# Save the model parameters
torch.save({
    'state_dict': lstm_model.state_dict(),
    'input_size': input_size,
    'hidden_size': hidden_size,
    'num_layers': num_layers,
    'output_size': output_size,
    
}, 'trained_lstm_model.pth')
        
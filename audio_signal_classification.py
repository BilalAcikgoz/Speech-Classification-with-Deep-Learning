import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import librosa
from scipy.io import wavfile as wav
from tqdm import tqdm
from scipy import ndimage
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn import metrics
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from warnings import filterwarnings
filterwarnings("ignore")

try:
    # if use GPU, use it
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {torch.cuda.get_device_name()} for training." if torch.cuda.is_available() else "Using CPU for training.")
except:
    print("No GPU found. Using CPU.")
    device = torch.device("cpu")

# We will extract MFCC's for every audio file in the dataset..
audio_dataset_path = 'UrbanSound8K/audio/'
metadata = pd.read_csv('UrbanSound8K/metadata/UrbanSound8K.csv')
def features_extractor(file_name):
    audio, sample_rate = librosa.load(file_name)
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=60)
    mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)

    return mfccs_scaled_features

# Now we iterate through every audio file and extract features - using MFCC
extracted_features=[]
for index_num,row in tqdm(metadata.iterrows()):
    file_name = os.path.join(os.path.abspath(audio_dataset_path),'fold'+str(row["fold"])+'/',str(row["slice_file_name"]))
    final_class_labels=row["class"]
    data=features_extractor(file_name)
    extracted_features.append([data,final_class_labels])

extracted_features_df = pd.DataFrame(extracted_features, columns=['feature', 'class'])

# Data augmentation
# Noise injection
def add_noise(audio_data, noise_factor):
    noise = np.random.randn(len(audio_data))
    augmented_data = audio_data + noise_factor * noise
    return augmented_data

# Apply data augmentation
def apply_augmentation(features_df, shift_max=5, noise_factor=0.005, num_augmented=5):
    augmented_features = []

    for index, row in features_df.iterrows():
        feature = row['feature']
        label = row['class']

        for i in range(num_augmented):
            augmented_feature = feature.copy()
            # add_noise
            augmented_data = add_noise(augmented_feature, noise_factor)
            # Append feature and label
            augmented_features.append([augmented_data, label])

    augmented_df = pd.DataFrame(augmented_features, columns=['feature', 'class'])
    return augmented_df

augmented_data = apply_augmentation(extracted_features_df)

x = np.array(augmented_data['feature'].tolist())
y = np.array(augmented_data['class'].tolist())

labelencoder = LabelEncoder()
y = to_categorical(labelencoder.fit_transform(y))

# train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=0)

x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
y_train_tensor = torch.tensor(np.argmax(y_train, axis=1), dtype=torch.long)

x_val_tensor = torch.tensor(x_val, dtype=torch.float32)
y_val_tensor = torch.tensor(np.argmax(y_val, axis=1), dtype=torch.long)

x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
y_test_tensor = torch.tensor(np.argmax(y_test, axis=1), dtype=torch.long)

train_dataset = torch.utils.data.TensorDataset(x_train_tensor, y_train_tensor)
val_dataset = torch.utils.data.TensorDataset(x_val_tensor, y_val_tensor)
test_dataset = torch.utils.data.TensorDataset(x_test_tensor, y_test_tensor)

batch_size = 80
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# LeNet
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 6, kernel_size=5, padding=2)
        self.pool1 = nn.AvgPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(6, 16, kernel_size=5)
        self.pool2 = nn.AvgPool1d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(16 * 13, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.sigmoid(self.conv1(x))
        x = self.pool1(x)
        x = self.sigmoid(self.conv2(x))
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.sigmoid(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        x = self.fc3(x)
        return torch.nn.functional.softmax(x, dim=1)

def train(model, train_loader, criterion, optimizer):
    model.train()
    total_loss = 0.0
    for batch_inputs, batch_labels in train_loader:
        batch_inputs, batch_labels = batch_inputs, batch_labels
        optimizer.zero_grad()
        outputs = model(batch_inputs)
        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    average_loss = total_loss / len(train_loader)
    return average_loss

def validate(model, val_loader, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_inputs, batch_labels in val_loader:
            batch_inputs, batch_labels = batch_inputs, batch_labels
            outputs = model(batch_inputs)
            loss = criterion(outputs, batch_labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += batch_labels.size(0)
            correct += (predicted == batch_labels).sum().item()
    accuracy = correct / total * 100
    average_loss = total_loss / len(val_loader)
    return accuracy, average_loss

# Creating Model
cnn_model = SimpleCNN()

# Loss function and optimizer identify
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(cnn_model.parameters(), lr=0.01)

# Train
num_epochs = 250
for epoch in range(num_epochs):
    train_loss = train(cnn_model, train_loader, criterion, optimizer)
    train_accuracy, _ = validate(cnn_model, train_loader, criterion)
    val_accuracy, _ = validate(cnn_model, val_loader, criterion)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Validation Accuracy: {val_accuracy:.2f}%')

# Train Accuracy: 95.76%, Validation Accuracy: 94.95%
torch.save(cnn_model, 'cnn_model.pth')
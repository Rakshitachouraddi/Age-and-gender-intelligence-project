import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
import numpy as np

# Step 1: Check GPU Availability
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

# Step 2: Dataset Loading and Preprocessing
def parse_filename(filename):
    parts = filename.split('_')
    age = int(parts[0])  # Age is the first part of the filename
    gender = int(parts[1])  # Gender is the second part (0: Male, 1: Female)
    return age, gender

class UTKFaceDataset(Dataset):
    def __init__(self, images, ages, genders, transform=None):
        self.images = images
        self.ages = ages
        self.genders = genders
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx].astype('float32')  # Convert to float32
        image = np.clip(image, 0, 1)  # Ensure values are in the range [0, 1]
        age = np.float32(self.ages[idx])  # Ensure float32
        gender = np.float32(self.genders[idx])  # Ensure float32

        if self.transform:
            image = self.transform(image)  # Apply transformations

        return image, torch.tensor(age, dtype=torch.float32), torch.tensor(gender, dtype=torch.float32)

def load_data(dataset_path, img_size=(128, 128)):
    images = []
    ages = []
    genders = []
    for file in os.listdir(dataset_path):
        if file.endswith('.jpg'):
            age, gender = parse_filename(file)
            img_path = os.path.join(dataset_path, file)
            img = cv2.imread(img_path)
            img = cv2.resize(img, img_size)
            images.append(img)
            ages.append(age)
            genders.append(gender)
    return np.array(images), np.array(ages), np.array(genders)

# Dataset Path (Replace with your dataset path)
dataset_path = 'UTKFace'

# Load dataset
img_size = (128, 128)
images, ages, genders = load_data(dataset_path, img_size)

# Normalize images and scale age
images = images.astype('float32') / 255.0  # Convert to float32 and normalize to [0, 1]
ages = ages.astype('float32') / 100.0      # Scale age to [0, 1] for regression

# Split into train and test sets
X_train, X_test, y_age_train, y_age_test, y_gender_train, y_gender_test = train_test_split(
    images, ages, genders, test_size=0.2, random_state=42
)

# Define image transformations
transform = transforms.Compose([
    transforms.ToTensor(),  # Converts to tensor and normalizes to [0, 1]
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize values to [-1, 1]
])

# Create PyTorch datasets
train_dataset = UTKFaceDataset(X_train, y_age_train, y_gender_train, transform=transform)
test_dataset = UTKFaceDataset(X_test, y_age_test, y_gender_test, transform=transform)

# Create PyTorch data loaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Step 3: Define the Model
class AgeGenderModel(nn.Module):
    def __init__(self):
        super(AgeGenderModel, self).__init__()
        self.shared_conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.flatten = nn.Flatten()
        self.shared_fc = nn.Sequential(
            nn.Linear(64 * 32 * 32, 128),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        # Branch for gender classification
        self.gender_fc = nn.Linear(128, 1)  # Sigmoid activation for binary classification
        # Branch for age regression
        self.age_fc = nn.Linear(128, 1)    # Linear activation for regression

    def forward(self, x):
        x = self.shared_conv(x)
        x = self.flatten(x)
        x = self.shared_fc(x)
        gender = torch.sigmoid(self.gender_fc(x))  # Gender output
        age = self.age_fc(x)  # Age output
        return gender, age

# Instantiate model and move to GPU
model = AgeGenderModel().to(device)

# Step 4: Define Loss Functions and Optimizer
gender_criterion = nn.BCELoss()  # Binary Cross-Entropy Loss for gender
age_criterion = nn.MSELoss()     # Mean Squared Error Loss for age
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Step 5: Train the Model
epochs = 20
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for images, ages, genders in train_loader:
        images, ages, genders = images.to(device), ages.to(device), genders.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        gender_pred, age_pred = model(images)

        # Compute losses
        gender_loss = gender_criterion(gender_pred, genders.unsqueeze(1))
        age_loss = age_criterion(age_pred, ages.unsqueeze(1))
        loss = gender_loss + age_loss

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(train_loader):.4f}")

# Step 6: Save the Trained Model
torch.save(model.state_dict(), 'age_gender_model_gpu.pth')

# Step 7: Real-Time Detection with Webcam
model.eval()

def preprocess_frame(frame, transform, img_size=(128, 128)):
    frame = cv2.resize(frame, img_size)
    frame = frame.astype('float32') / 255.0  # Normalize
    frame = transform(frame).unsqueeze(0)  # Add batch dimension
    return frame

# Load trained model weights
model.load_state_dict(torch.load('age_gender_model_gpu.pth'))


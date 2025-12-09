import glob

import os
import json
from PIL import Image
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
from torchvision import transforms, models

import matplotlib.pyplot as plt

import pandas as pd


data_path = "../../../data"
pseudo_mixtures_paths = glob.glob(f"{data_path}/Pseudo_Mixtures/*.jpg")

labels = {}

df = pd.read_csv('../../data/Pseudo_Mixtures/ratios.csv', sep=",")

for index, row in df.iterrows():
    labels[row['path']] = [float(row['spratRatio']), float(row['herringRatio']), float(row['smelt']), float(row['stickleback'])]
    
# 2. Custom dataset
class RatioDataset(Dataset):
    def __init__(self, labels, transform=None):
        self.items = list(labels.items())
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img_path, ratio_vector = self.items[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        
        ratio_vector = torch.tensor(ratio_vector, dtype=torch.float32)    
        
        return image, ratio_vector

# 3. Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# 4. Dataset split & DataLoader
dataset = RatioDataset(labels, transform)

val_size = int(0.2 * len(dataset))
train_size = len(dataset) - val_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=16, shuffle=False)


# 5. Model (ResNet backbone)
model = models.resnet18(pretrained=True)
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 128),
    nn.ReLU(),
    nn.Linear(128, 4),     # output 4 values
    nn.Softmax(dim=1)      # ensure they sum to 1
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 6. Training setup
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# 7. Training loop
num_epochs = 10
train_losses = []
val_losses = []

for epoch in range(num_epochs):
    # --- TRAINING ---
    model.train()
    running_train_loss = 0.0
    
    for imgs, ratios in train_loader:
        imgs, ratios = imgs.to(device), ratios.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, ratios)
        loss.backward()
        optimizer.step()

        running_train_loss += loss.item() * imgs.size(0)

    epoch_train_loss = running_train_loss / train_size
    train_losses.append(epoch_train_loss)

    # --- VALIDATION ---
    model.eval()
    running_val_loss = 0.0

    with torch.no_grad():
        for imgs, ratios in val_loader:
            imgs, ratios = imgs.to(device), ratios.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, ratios)
            running_val_loss += loss.item() * imgs.size(0)

    epoch_val_loss = running_val_loss / val_size
    val_losses.append(epoch_val_loss)

    print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f}")

torch.save(model.state_dict(), "ratio_model.pth")
print("Model saved!")

# 8. Performance Graph
plt.figure(figsize=(8,5))

plt.plot(train_losses, label="Train Loss", linewidth=3, color="#f45d4a")
plt.plot(val_losses, label="Validation Loss", linewidth=3, color="#494096")

plt.xlabel("Epochs", fontsize=12)
plt.ylabel("Loss", fontsize=12)
plt.title("Training vs Validation Loss", fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=12)

plt.tight_layout()
plt.show()

import glob

import os
import json
from PIL import Image
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

data_path = "../../data"
pseudo_mixtures_paths = glob.glob(f"{data_path}/Pseudo_Mixtures/*.jpg")

labels = {}

for path in pseudo_mixtures_paths:
    ratio = path.split(".jpg")[-2].split("_")[-1]
    labels[path] = float(ratio)
    
# 2. Custom dataset
class RatioDataset(Dataset):
    def __init__(self, labels, transform=None):
        self.items = list(labels.items())
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img_path, ratio = self.items[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor([ratio], dtype=torch.float32)

# 3. Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# 4. Dataset & DataLoader
dataset = RatioDataset(labels, transform)
train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

# 5. Model (ResNet backbone)
model = models.resnet18(pretrained=True)
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 128),
    nn.ReLU(),
    nn.Linear(128, 1),
    nn.Sigmoid()  # output between 0 and 1
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 6. Training setup
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# 7. Training loop
for epoch in range(10):
    model.train()
    running_loss = 0.0
    for imgs, ratios in train_loader:
        imgs, ratios = imgs.to(device), ratios.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, ratios)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)

    epoch_loss = running_loss / len(dataset)
    print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}")

torch.save(model.state_dict(), "ratio_model.pth")
print("Model saved!")

def predict_ratio(model, image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        ratio = model(image).item()
    return ratio

#ratio = predict_ratio(model, "test_image.jpg")
#print(f"Predicted ratio of item A: {ratio:.2f}") 
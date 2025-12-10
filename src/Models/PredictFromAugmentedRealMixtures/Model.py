import glob
import os
from PIL import Image

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models

import matplotlib.pyplot as plt


# -----------------------
# PATHS
# -----------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# PredictFromAugmentedRealMixtures -> Models -> src -> PROJECT ROOT
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", "..", ".."))

DATA_PATH = os.path.join(PROJECT_ROOT, "data")

# TRAINING ONLY
TRAIN_IMAGE_FOLDER = os.path.join(
    DATA_PATH,
    "Real_Mixtures_Augmented",
    "training"
)

# MODEL SAVE PATH
MODEL_SAVE_PATH = os.path.join(
    PROJECT_ROOT,
    "src", "Models", "ModelFiles",
    "real_augmented_model.pth"
)

print("PROJECT ROOT:", PROJECT_ROOT)
print("TRAIN IMAGE FOLDER:", TRAIN_IMAGE_FOLDER)
print("MODEL SAVE PATH:", MODEL_SAVE_PATH)


# -----------------------
# LOAD TRAINING IMAGES ONLY
# -----------------------
image_paths = (
    glob.glob(os.path.join(TRAIN_IMAGE_FOLDER, "*.png")) +
    glob.glob(os.path.join(TRAIN_IMAGE_FOLDER, "*.jpg")) +
    glob.glob(os.path.join(TRAIN_IMAGE_FOLDER, "*.jpeg"))
)

print("TRAINING IMAGES FOUND:", len(image_paths))

if len(image_paths) == 0:
    raise RuntimeError("No images found in training folder.")


# -----------------------
# LABEL EXTRACTION FROM FILENAME
# Expected format:
#   something_0.65_aug_3.png
# -----------------------
def extract_ratio_from_filename(path):
    name = os.path.splitext(os.path.basename(path))[0]
    parts = name.split("_")

    for token in reversed(parts):
        try:
            return float(token)
        except ValueError:
            continue

    raise ValueError(f"Could not extract ratio from: {path}")


labels = {}
skipped = 0

for path in image_paths:
    try:
        ratio = extract_ratio_from_filename(path)
        labels[path] = float(ratio)
    except ValueError:
        skipped += 1

print("VALID LABELED TRAIN IMAGES:", len(labels))
print("SKIPPED (NO RATIO IN NAME):", skipped)

if len(labels) == 0:
    raise RuntimeError("No valid labeled images found in training folder.")


# -----------------------
# DATASET
# -----------------------
class RatioDataset(Dataset):
    def __init__(self, labels_dict, transform=None):
        self.items = list(labels_dict.items())
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img_path, ratio = self.items[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        ratio = torch.tensor([ratio], dtype=torch.float32)
        return image, ratio


# -----------------------
# TRANSFORMS (SAME AS INFERENCE)
# -----------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])


# -----------------------
# DATASET SPLIT (TRAIN / VAL ONLY)
# -----------------------
dataset = RatioDataset(labels, transform)

val_size = int(0.2 * len(dataset))
train_size = len(dataset) - val_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=16, shuffle=False)

print("TRAIN SAMPLES:", train_size)
print("VAL SAMPLES  :", val_size)


# -----------------------
# MODEL (RESNET18 REGRESSION)
# -----------------------
model = models.resnet18(pretrained=True)

model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 128),
    nn.ReLU(),
    nn.Linear(128, 1),
    nn.Sigmoid()
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print("TRAINING ON DEVICE:", device)


# -----------------------
# TRAINING SETUP
# -----------------------
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)


# -----------------------
# TRAINING LOOP
# -----------------------
num_epochs = 10
train_losses = []
val_losses = []

for epoch in range(num_epochs):

    # ----- TRAIN -----
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


    # ----- VALIDATION -----
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

    print(
        f"Epoch {epoch+1}/{num_epochs} | "
        f"Train Loss: {epoch_train_loss:.4f} | "
        f"Val Loss: {epoch_val_loss:.4f}"
    )


# -----------------------
# SAVE MODEL
# -----------------------
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

torch.save(model.state_dict(), MODEL_SAVE_PATH)
print("MODEL SAVED TO:", MODEL_SAVE_PATH)


# -----------------------
# LOSS CURVE
# -----------------------
plt.figure(figsize=(8, 5))

plt.plot(train_losses, label="Train Loss", linewidth=3)
plt.plot(val_losses, label="Validation Loss", linewidth=3)

plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()
import glob
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import os
from matplotlib import pyplot as plt

# =============================================================================
# paths
# =============================================================================
data_path = "../../../data"  # Relative path from this script to data folder
dataset_path = f"{data_path}/SingleFishes/SingleFishes/finalSFdataset"  # Your dataset folder

# =============================================================================
# Dataset creation
# =============================================================================
class FishDataset(Dataset):
    """
    Custom PyTorch Dataset for loading fish images
    
    folder structure:
        finalSFdataset/
            herring/
                fish1.jpg
                fish2.jpg
            sprat/
                fish1.jpg
                fish2.jpg
    """
    
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (str): Path to dataset folder (contains herring/ and sprat/)
            transform (callable, optional): Transformations to apply to images
        """
        self.root_dir = root_dir
        self.transform = transform
        
        # Define classes (folder names become labels)
        self.classes = ['herring', 'sprat']
        self.class_to_idx = {'herring': 0, 'sprat': 1}  # Convert names to numbers
        
        # Storage for image paths and labels
        self.images = []  # List of image file paths
        self.labels = []  # List of corresponding labels (0 or 1)
        
        # Scan each class folder for images
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)  # e.g., finalSFdataset/herring
            
            if os.path.exists(class_dir):
                # Find all image files (jpg, jpeg, png)
                image_files = []
                image_files += glob.glob(os.path.join(class_dir, "*.jpg"))
                image_files += glob.glob(os.path.join(class_dir, "*.jpeg"))
                image_files += glob.glob(os.path.join(class_dir, "*.png"))
                
                # Add each image and its label
                for img_path in image_files:
                    self.images.append(img_path)
                    self.labels.append(self.class_to_idx[class_name])
                
                print(f"Found {len(image_files)} {class_name} images")
            else:
                print(f"WARNING: Folder '{class_dir}' does not exist!")
        
        print(f"\nTotal: {len(self.images)} images loaded")
        print(f"  Herring (0): {self.labels.count(0)}")
        print(f"  Sprat (1): {self.labels.count(1)}")
    
    def __len__(self):
        """Returns the total number of images"""
        return len(self.images)
    
    def __getitem__(self, idx):
        """
        Load and return a single image and its label
        
        Args:
            idx (int): Index of the image to load
            
        Returns:
            tuple: (image, label) where image is a tensor and label is 0 or 1
        """
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # Load image from disk
        image = Image.open(img_path).convert('RGB')  # Ensure RGB format
        
        # Apply transformations if specified
        if self.transform:
            image = self.transform(image)
        
        return image, label

# =============================================================================
# Image transformations parameters
# =============================================================================

# Transformations for TRAINING (includes data augmentation)
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),                          # Resize to 224x224
    transforms.RandomHorizontalFlip(p=0.5),                 # Flip 50% of images
    transforms.RandomRotation(degrees=15),                  # Rotate ±15 degrees
    transforms.ColorJitter(brightness=0.2, contrast=0.2),   # Vary brightness/contrast
    transforms.ToTensor(),                                  # Convert to PyTorch tensor
    transforms.Normalize(                                   # Normalize for pretrained models
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Transformations for VALIDATION/TESTING (no augmentation)
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# =============================================================================
# Checkup to see if we can load the data
# =============================================================================
if __name__ == "__main__":
    print("="*70)
    print("LOADING DATASET")
    print("="*70)
    
    # Create dataset
    dataset = FishDataset(dataset_path, transform=train_transform)
    
    # Check if dataset is empty
    if len(dataset) == 0:
        print("\n❌ ERROR: No images found!")
        print("\nMake sure your folder structure looks like this:")
        print("  data/SingleFishes/finalSFdataset/")
        print("    ├── herring/")
        print("    │   └── (your herring .jpg/.png images)")
        print("    └── sprat/")
        print("        └── (your sprat .jpg/.png images)")
        exit(1)
    
    print("\n" + "="*70)
    print("CREATING DATA LOADERS")
    print("="*70)
    
    # Split into train (80%) and validation (20%)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    print(f"Training set: {train_size} images")
    print(f"Validation set: {val_size} images")
    
    # Create data loaders (for batching during training)
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,      # Load 32 images at a time
        shuffle=True,       # Shuffle training data
        num_workers=0       # No parallel workers (Windows compatible)
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,      # Don't shuffle validation data
        num_workers=0
    )
    
    print("\n" + "="*70)
    print("TESTING DATA LOADING")
    print("="*70)
    
    # Load one batch to test
    images, labels = next(iter(train_loader))
    
    print(f"✓ Batch loaded successfully!")
    print(f"  Batch shape: {images.shape}")  # Should be [32, 3, 224, 224]
    print(f"  Labels shape: {labels.shape}")  # Should be [32]
    print(f"  Sample labels: {labels[:10].tolist()}")  # First 10 labels
    print(f"  Label distribution in batch:")
    print(f"    Herring (0): {(labels == 0).sum().item()}")
    print(f"    Sprat (1): {(labels == 1).sum().item()}")
    
    print("\n✅ Dataset setup complete! Ready for model training.")



# =============================================================================
# Model training 
# =============================================================================

model = models.resnet18(pretrained=True)
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 128),
    nn.ReLU(),
    nn.Linear(128, 2),     # output 2 values
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load previous training if it exists
checkpoint_path = "fish_classifier_checkpoint.pth"
start_epoch = 0
train_losses = []
val_losses = []
train_accs = []
val_accs = []

if os.path.exists(checkpoint_path):
    print(f"Loading saved checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    train_losses = checkpoint.get('train_losses', [])
    val_losses = checkpoint.get('val_losses', [])
    train_accs = checkpoint.get('train_accs', [])
    val_accs = checkpoint.get('val_accs', [])
    start_epoch = checkpoint.get('epoch', 0)
    print(f"Resuming from epoch {start_epoch}")
    print(f"Previous best val acc: {max(val_accs):.2f}%" if val_accs else "")
    response = input("Continue training from checkpoint? (yes/no): ").strip().lower()
    if response not in ['yes', 'y']:
        print("Starting fresh training...")
        train_losses = []
        val_losses = []
        train_accs = []
        val_accs = []
        start_epoch = 0
else:
    print("No checkpoint found. Starting fresh training...")

# 6. Training setup
criterion = nn.CrossEntropyLoss()  # For classification tasks
optimizer = optim.Adam(model.parameters(), lr=1e-5)

# 7. Training loop
num_epochs = 20  # This will train for 20 MORE epochs

for epoch in range(start_epoch, start_epoch + num_epochs):
    # --- TRAINING ---
    model.train()
    running_train_loss = 0.0
    correct_train = 0
    total_train = 0
    
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_train_loss += loss.item() * imgs.size(0)
        
        # Calculate accuracy
        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

    epoch_train_loss = running_train_loss / train_size
    epoch_train_acc = 100 * correct_train / total_train
    train_losses.append(epoch_train_loss)
    train_accs.append(epoch_train_acc)

    # --- VALIDATION ---
    model.eval()
    running_val_loss = 0.0
    correct_val = 0
    total_val = 0

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            running_val_loss += loss.item() * imgs.size(0)
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()

    epoch_val_loss = running_val_loss / val_size
    epoch_val_acc = 100 * correct_val / total_val
    val_losses.append(epoch_val_loss)
    val_accs.append(epoch_val_acc)

    print(f"Epoch {epoch+1}/{start_epoch + num_epochs} | Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | Train Acc: {epoch_train_acc:.2f}% | Val Acc: {epoch_val_acc:.2f}%")

# Save full checkpoint with training history
checkpoint = {
    'epoch': epoch + 1,
    'model_state_dict': model.state_dict(),
    'train_losses': train_losses,
    'val_losses': val_losses,
    'train_accs': train_accs,
    'val_accs': val_accs
}
torch.save(checkpoint, "fish_classifier_checkpoint.pth")
print(f"Checkpoint saved! (Epoch {epoch+1})")

# 8. Performance Graphs
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Loss plot
ax1.plot(train_losses, label="Train Loss", linewidth=3, color="#f45d4a")
ax1.plot(val_losses, label="Validation Loss", linewidth=3, color="#494096")
ax1.set_xlabel("Epochs", fontsize=12)
ax1.set_ylabel("Loss", fontsize=12)
ax1.set_title("Training vs Validation Loss", fontsize=14)
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=12)

# Accuracy plot
ax2.plot(train_accs, label="Train Accuracy", linewidth=3, color="#f45d4a")
ax2.plot(val_accs, label="Validation Accuracy", linewidth=3, color="#494096")
ax2.set_xlabel("Epochs", fontsize=12)
ax2.set_ylabel("Accuracy (%)", fontsize=12)
ax2.set_title("Training vs Validation Accuracy", fontsize=14)
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=12)

plt.tight_layout()
plt.show()

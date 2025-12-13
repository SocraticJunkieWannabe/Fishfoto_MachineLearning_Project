"""
Evaluate trained model and display:
- Confusion matrix
- Training/validation loss curves
- Training/validation accuracy curves
"""

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms, models
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import os
import glob
from PIL import Image

# =============================================================================
# Configuration
# =============================================================================
data_path = "../../../data"
dataset_path = f"{data_path}/SingleFishes/SingleFishes/finalSFdataset"
checkpoint_path = "fish_classifier_checkpoint.pth"

# =============================================================================
# Dataset class (same as Model.py)
# =============================================================================
class FishDataset:
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = ['herring', 'sprat']
        self.class_to_idx = {'herring': 0, 'sprat': 1}
        self.images = []
        self.labels = []
        
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            if os.path.exists(class_dir):
                image_files = []
                image_files += glob.glob(os.path.join(class_dir, "*.jpg"))
                image_files += glob.glob(os.path.join(class_dir, "*.jpeg"))
                image_files += glob.glob(os.path.join(class_dir, "*.png"))
                
                for img_path in image_files:
                    self.images.append(img_path)
                    self.labels.append(self.class_to_idx[class_name])
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# =============================================================================
# Main evaluation
# =============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("MODEL EVALUATION AND VISUALIZATION")
    print("=" * 70)
    
    # Check if checkpoint exists
    if not os.path.exists(checkpoint_path):
        print(f"❌ Error: Checkpoint not found at {checkpoint_path}")
        print("Please train the model first using Model.py")
        exit(1)
    
    # Load checkpoint
    print(f"\nLoading checkpoint from {checkpoint_path}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load training history
    train_losses = checkpoint.get('train_losses', [])
    val_losses = checkpoint.get('val_losses', [])
    train_accs = checkpoint.get('train_accs', [])
    val_accs = checkpoint.get('val_accs', [])
    epochs_trained = checkpoint.get('epoch', 0)
    
    print(f"✓ Loaded checkpoint from epoch {epochs_trained}")
    print(f"✓ Training history: {len(train_losses)} epochs")
    
    # Recreate model architecture
    model = models.resnet18(pretrained=False)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 128),
        nn.ReLU(),
        nn.Linear(128, 2)
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print("✓ Model loaded successfully")
    
    # Prepare data
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    dataset = FishDataset(dataset_path, transform=val_transform)
    
    # Use same split as training (80/20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # Same seed for reproducibility
    )
    
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    print(f"✓ Loaded {val_size} validation images")
    
    # =============================================================================
    # Generate predictions for confusion matrix
    # =============================================================================
    print("\nGenerating predictions...")
    
    all_labels = []
    all_predictions = []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
    
    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)
    
    # =============================================================================
    # Plot 1: Confusion Matrix
    # =============================================================================
    cm = confusion_matrix(all_labels, all_predictions)
    
    # Calculate metrics
    accuracy = 100 * np.sum(all_labels == all_predictions) / len(all_labels)
    
    print("\n" + "=" * 70)
    print("CLASSIFICATION REPORT")
    print("=" * 70)
    print(classification_report(all_labels, all_predictions, 
                                target_names=['Herring', 'Sprat']))
    
    # =============================================================================
    # Create visualizations
    # =============================================================================
    fig = plt.figure(figsize=(16, 5))
    
    # Subplot 1: Confusion Matrix
    ax1 = plt.subplot(1, 3, 1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Herring', 'Sprat'],
                yticklabels=['Herring', 'Sprat'],
                cbar_kws={'label': 'Count'})
    ax1.set_xlabel('Predicted', fontsize=12)
    ax1.set_ylabel('Actual', fontsize=12)
    ax1.set_title(f'Confusion Matrix\nAccuracy: {accuracy:.2f}%', fontsize=14, fontweight='bold')
    
    # Subplot 2: Loss curves
    ax2 = plt.subplot(1, 3, 2)
    epochs = range(1, len(train_losses) + 1)
    ax2.plot(epochs, train_losses, label='Train Loss', linewidth=3, color='#f45d4a', marker='o', markersize=4)
    ax2.plot(epochs, val_losses, label='Validation Loss', linewidth=3, color='#494096', marker='s', markersize=4)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.set_title('Training vs Validation Loss', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    
    # Subplot 3: Accuracy curves
    ax3 = plt.subplot(1, 3, 3)
    ax3.plot(epochs, train_accs, label='Train Accuracy', linewidth=3, color='#f45d4a', marker='o', markersize=4)
    ax3.plot(epochs, val_accs, label='Validation Accuracy', linewidth=3, color='#494096', marker='s', markersize=4)
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('Accuracy (%)', fontsize=12)
    ax3.set_title('Training vs Validation Accuracy', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=10)
    ax3.set_ylim([0, 105])
    
    plt.tight_layout()
    
    # Save figure
    save_path = "model_evaluation.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Visualization saved to {save_path}")
    
    plt.show()
    
    print("\n" + "=" * 70)
    print(f"FINAL VALIDATION ACCURACY: {accuracy:.2f}%")
    print(f"Best Validation Accuracy: {max(val_accs):.2f}% (Epoch {val_accs.index(max(val_accs)) + 1})")
    print("=" * 70)

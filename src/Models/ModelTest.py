import torch
from torchvision import models
from torch import nn
from PIL import Image
from torchvision import transforms

#FROM CNN#
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1️⃣ Define the model architecture (same as during training)
model = models.resnet18(pretrained=False)
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 128),
    nn.ReLU(),
    nn.Linear(128, 1),
    nn.Sigmoid()  # output between 0 and 1
)

# 2️⃣ Load saved weights
model.load_state_dict(torch.load("ratio_model.pth", map_location=device))
model.to(device)
model.eval()
print("✅ Model loaded successfully!")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def predict_ratio(model, image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        ratio = model(image).item()
    return ratio

ratio = predict_ratio(model, "test_crop.jpg")
print(f"Predicted ratio of sprat: {ratio:.2f}")
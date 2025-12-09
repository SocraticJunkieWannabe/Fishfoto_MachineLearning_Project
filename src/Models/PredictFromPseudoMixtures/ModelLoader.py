import torch
from torch import nn
from torchvision import transforms, models


def loadModel():
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
    model.load_state_dict(torch.load("model.pth", map_location=device))
    model.to(device)
    model.eval()
    print("✅ Model loaded successfully!")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                            [0.229, 0.224, 0.225])
    ])
    
    return (device, model, transform)
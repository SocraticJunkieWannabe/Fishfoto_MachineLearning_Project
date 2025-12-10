import torch
from torch import nn
from torchvision import transforms, models


def loadModel(model_file_path):
    """
    Loads a trained PyTorch regression model that predicts a ratio in [0, 1]
    from an input image.

    Output:
        device   : torch.device  -> CPU or GPU used for inference
        model    : torch.nn.Module -> ResNet18 model with loaded weights
        transform: torchvision.transforms.Compose -> image preprocessing pipeline

    The model outputs:
        - A single float value in the range [0, 1]
        - This represents a continuous ratio (regression), not a class
          Example: 0.65 means 65% mixture of a given component
    """

    # 1️⃣ Select device (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2️⃣ Define the model architecture (MUST match training exactly)
    model = models.resnet18(pretrained=False)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 128),
        nn.ReLU(),
        nn.Linear(128, 1),
        nn.Sigmoid()  # Output is a single value between 0 and 1 (regression)
    )

    # 3️⃣ Load saved weights
    model.load_state_dict(torch.load(model_file_path, map_location=device))
    model.to(device)
    model.eval()

    print("Model loaded successfully from:", model_file_path)

    # 4️⃣ Define the EXACT same transform as used during training
    transform = transforms.Compose([
        transforms.Resize((224, 224)),   # ResNet input size
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],   # ImageNet normalization
            std=[0.229, 0.224, 0.225]
        )
    ])

    # 5️⃣ Return everything needed for inference
    return (device, model, transform)
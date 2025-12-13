import torch
from torch import nn
from torchvision import transforms, models
import os


def loadModel(model_file_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1️⃣ Define the model architecture (same as during training)
    model = models.resnet18(pretrained=False)  # No need for pretrained when loading weights
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 128),
        nn.ReLU(),
        nn.Linear(128, 2),  # 2 classes: herring (0) and sprat (1)
    )

    # 2️⃣ Load checkpoint
    print(f"Loading model from {model_file_path}...")
    checkpoint = torch.load(model_file_path, map_location=device)
    
    # Handle both checkpoint format and simple state_dict format
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint.get('epoch', 'unknown')
        print(f"✅ Model loaded successfully from epoch {epoch}!")
    else:
        # Legacy format (just state_dict)
        model.load_state_dict(checkpoint)
        print("✅ Model loaded successfully!")
    
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


def predict_single_image(image, device, model, transform):
    """
    Predict the class of a single fish image.
    
    Args:
        image: PIL Image object
        device: torch device
        model: loaded model
        transform: image preprocessing transform
        
    Returns:
        tuple: (predicted_class, confidence, probabilities)
            - predicted_class (int): 0 for herring, 1 for sprat
            - confidence (float): probability of predicted class (0-1)
            - probabilities (dict): {'herring': prob, 'sprat': prob}
    """
    # Preprocess image
    img_tensor = transform(image).unsqueeze(0).to(device)
    
    # Get prediction
    with torch.no_grad():
        output = model(img_tensor)
        probabilities = torch.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    predicted_class = predicted.item()
    confidence_value = confidence.item()
    
    # Create probability dictionary
    probs = probabilities[0].cpu().numpy()
    prob_dict = {
        'herring': float(probs[0]),
        'sprat': float(probs[1])
    }
    
    return predicted_class, confidence_value, prob_dict


from PIL import Image
from ModelLoader import loadModel, predict_single_image

# Load model
device, model, transform = loadModel("fish_classifier_checkpoint.pth")

# Predict
img = Image.open("my_fish.jpg")
cls, conf, probs = predict_single_image(img, device, model, transform)

print(f"Class: {'Herring' if cls == 0 else 'Sprat'}")
print(f"Confidence: {conf*100:.1f}%")
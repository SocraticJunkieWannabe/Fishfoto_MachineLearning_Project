import torch
from torchvision import models
from torch import nn
from PIL import Image
from torchvision import transforms

import pandas as pd
import os
import numpy as np

df = pd.read_csv('../../data/percentages.csv', sep=";")
print(df)

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

def testAgainstAllStockMixtures():
    mixtures_images_path = "../../data/Labeled Images/mixture"
    
    folderContent = os.listdir(mixtures_images_path)
    ratios = []
    
    for fileName in folderContent:
        print(f"testing model on {fileName}")
        output = testModelOnImage(f"{mixtures_images_path}/{fileName}")
        if output != None:
            ratios.append(output)
        
    print(computeError(ratios))
    
    pass


def testModelOnImage(image_path):
    
    #True Ratio Extraction
    image_file_name = image_path.split("/")[-1]
    haulNumber = image_file_name.split("_")[0].split("T")[1]
    
    if haulNumber not in df.columns:
        return None

    predicted_ratio = predict_ratio(model, image_path)

    ratio_from_df = str(df.loc[1, haulNumber]).split(".")[0]
    actual_ratio = float(f"0.{ratio_from_df}")

    #Visual Output
    print(f"Predicted ratio of sprat: {predicted_ratio:.2f}")
    print(f"Actual ratio is {actual_ratio}")
    
    return predicted_ratio, actual_ratio

def computeError(data):
    #MSE
    error = 0
    for item in data:
        error += np.sqrt(np.square(item[0]-item[1]))
        
    error = error/len(data)
    return error

image_file_name = "T1_proov_20241014_065408841.jpg"
testAgainstAllStockMixtures()
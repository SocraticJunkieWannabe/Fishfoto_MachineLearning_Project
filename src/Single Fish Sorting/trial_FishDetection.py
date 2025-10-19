from ultralytics import YOLO
import cv2
import os
import numpy as np

import Sorters  


def extractFishesFromImages(sorterType : str, input_folder: str, single_output_folder : str, bundle_output_folder : str):
    # 2. Input and output folders
    images_path = "../../data"
    input_folder = f"{images_path}/{input_folder}"
    output_single_folder = f"{images_path}/{single_output_folder}"
    output_bundle_folder = f"{images_path}/{bundle_output_folder}"
    
    os.makedirs(output_single_folder, exist_ok=True)
    os.makedirs(output_bundle_folder, exist_ok=True)
    
    sorter = None
    
    if sorterType == "shape":
        sorter = Sorters.ShapeSorter(model, output_single_folder, output_bundle_folder)
    elif sorterType == "box":
        sorter = Sorters.BoxSorter(model, output_single_folder, output_bundle_folder)

    # 3. Loop over all images
    for image_name in os.listdir(input_folder):
        print(image_name)
        if not image_name.lower().endswith(('.jpg', '.png', '.jpeg')):
            continue
        img_path = os.path.join(input_folder, image_name)
        results = model(img_path,
                        iou=0.5,         # Allow more overlapping detections
                        imgsz=1280,      # Larger input image size → better small-object detection
                        max_det=300,      # Increase number of detections per image
                        )

        # 4. For each detection, crop the fish and save it
        #extractImageBoxFromPrediciton(results, img_path, image_name, output_single_folder)
        #extractImageShapeFromPrediciton(results, img_path, output_bundle_folder)
        sorter.extractImageFromPrediciton(results, img_path, image_name)
        
    pass


# 1. Load a pretrained YOLO model (try the large one for better accuracy)
model = YOLO("yolo11x-seg.pt")  

sorterType = "shape"

folder_output_single = f"Tests/{sorterType}_fish_crops"

extractFishesFromImages(sorterType, "Tests/test_raw_images", folder_output_single, "Tests/test_bundle_crops")

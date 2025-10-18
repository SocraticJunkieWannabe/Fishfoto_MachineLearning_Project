from ultralytics import YOLO
import cv2
import os
import numpy as np

import Sorters

def enhance_contrast(img):
    # Convert to LAB color space to adjust contrast
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.equalizeHist(l)
    enhanced = cv2.merge((l, a, b))
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    # Optional: slight sharpening
    kernel = np.array([[0, -1, 0],
                       [-1, 5,-1],
                       [0, -1, 0]])
    enhanced = cv2.filter2D(enhanced, -1, kernel)
    return enhanced

def getBoxImage(box, img):
    
    xyxy = box.xyxy[0].cpu().numpy().astype(int)
    x1, y1, x2, y2 = xyxy
        
    boxImage = img[y1:y2, x1:x2]
    
    return boxImage

def extractFishBoxImageFromBundle(img, image_name, output_folder : str, bundle_id : int):
    results = model.predict(img, 
                            imgsz=1280, 
                            conf=0.3, 
                            iou=0.5,
                            max_det=150
                            )
    
    
    
    for i, box in enumerate(results[0].boxes):
        cls = int(box.cls)
        conf = float(box.conf)

        # Only save if confidence is decent
        if conf > 0.1:
            crop = getBoxImage(box, img)

            crop_name = f"{os.path.splitext(image_name)[0]}_from-bundle-{bundle_id}_fish{i}.jpg"
            
            imageExists = checkIfImageDuplicateExists(crop)
            if not imageExists:
                cv2.imwrite(os.path.join(output_folder, crop_name), crop)
                save_images.append(crop)

def checkIfImageDuplicateExists(input_img, thershold : float = 0.1):
    return False

def extractImageBoxFromPrediciton(results, img_path, image_name, output_single_folder):
    for i, box in enumerate(results[0].boxes):
            cls = int(box.cls)
            conf = float(box.conf)

            # Only save if confidence is decent
            if conf > 0.1:
                img = cv2.imread(img_path)
                
                crop = getBoxImage(box, img)

                crop_name = f"{os.path.splitext(image_name)[0]}_fish{i}.jpg"
                
                imageExists = checkIfImageDuplicateExists(crop)
                
                if not imageExists:
                    if crop.shape[0] > 1500:
                        #process to extract single from bundle
                        extractFishBoxImageFromBundle(crop, image_name, output_single_folder, i)
                    else:
                        #save
                        if not os.path.isfile(crop_name):
                            cv2.imwrite(os.path.join(output_single_folder, crop_name), crop)
                            save_images.append(crop)
          

# 1. Load a pretrained YOLO model (try the large one for better accuracy)
model = YOLO("yolov8x-seg.pt")  # or yolov11x.pt if available

images_path = "../../data"

save_images = []

def splitLabeledImages():
    
    extractFishesFromImages()
    
    pass

def extractFishesFromImages(input_folder: str, single_output_folder : str, bundle_output_folder : str):
    # 2. Input and output folders
    input_folder = f"{images_path}/{input_folder}"
    output_single_folder = f"{images_path}/{single_output_folder}"
    output_bundle_folder = f"{images_path}/{bundle_output_folder}"
    
    os.makedirs(output_single_folder, exist_ok=True)
    os.makedirs(output_bundle_folder, exist_ok=True)
    
    shapeSorter = Sorters.ShapeSorter(model, output_single_folder, output_bundle_folder)

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
        shapeSorter.extractImageFromPrediciton(results, img_path)
        
    pass



extractFishesFromImages("Tests/test_raw_images", "Tests/test_fish_crops", "Tests/test_bundle_crops")

from ultralytics import YOLO
import cv2
import os
import numpy as np


class Sorter():
    
    def __init__(self, model, output_single_folder, output_bundle_folder, DEBUG_FLAG : bool = False):
        self.model = model
        self.save_images = []
        self.output_single_folder = output_single_folder
        self.output_bundle_folder = output_bundle_folder
        pass
    
    def extractFishImageFromBundle(self):
        pass
    
    def extractImageFromPrediciton(self):
        pass
    
    def checkIfImageDuplicateExists(self, input_img, thershold : float = 0.1):
        return False
    
    def getBoxImage(self, box, img):
    
        xyxy = box.xyxy[0].cpu().numpy().astype(int)
        x1, y1, x2, y2 = xyxy
            
        boxImage = img[y1:y2, x1:x2]
        
        return boxImage
    
class ShapeSorter(Sorter):
    
    def extractFishImageFromBundle(self, img, bundle_id : int):
        results = self.model.predict(img, 
                                imgsz=1280, 
                                conf=0.3, 
                                iou=0.5,
                                max_det=150
                                )
        
        orig_h, orig_w = img.shape[:2]
        
        for i, result in enumerate(results):
            if result.masks is not None:
                masks = result.masks.data.cpu().numpy()  # shape: (N, H_model, W_model)
                
                for j, mask in enumerate(masks):
                    # Resize mask to original image size
                    mask_resized = cv2.resize(mask, (orig_w, orig_h))
                    mask_uint8 = (mask_resized * 255).astype(np.uint8)

                    # Create RGBA image
                    rgba = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
                    rgba[:, :, 3] = mask_uint8  # alpha = mask

                    cv2.imwrite(f"{self.output_single_folder}/bundle-{bundle_id}_object_{i}_{j}.png", rgba)
        pass
    
    def extractImageFromPrediciton(self, results, img_path, image_name):
        orig = cv2.imread(img_path)
        orig_h, orig_w = orig.shape[:2]

        for i, result in enumerate(results):
            if result.masks is not None:
                masks = result.masks.data.cpu().numpy()  # shape: (N, H_model, W_model)
                
                for j, mask in enumerate(masks):
                    # Resize mask to original image size
                    mask_resized = cv2.resize(mask, (orig_w, orig_h))
                    mask_uint8 = (mask_resized * 255).astype(np.uint8)

                    # Create RGBA image
                    rgba = cv2.cvtColor(orig, cv2.COLOR_BGR2BGRA)
                    rgba[:, :, 3] = mask_uint8  # alpha = mask
                    
                    box = results[i].boxes[j]
                    crop = super().getBoxImage(box, orig)
                    
                    if crop.shape[0] > 1500 or crop.shape[1] > 1500:
                        #process to extract single from bundle
                        cv2.imwrite(f"{self.output_single_folder}/test.png", crop)
                        self.extractFishImageFromBundle(crop, i)
                    else:
                        cv2.imwrite(f"{self.output_single_folder}/object_{i}_{j}.png", super().getBoxImage(box, rgba))
        pass
    
class BoxSorter(Sorter):
    
    def extractFishImageFromBundle(self, img, image_name, bundle_id : int):
        results = self.model.predict(img, 
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
                crop = super().getBoxImage(box, img)

                crop_name = f"{os.path.splitext(image_name)[0]}_from-bundle-{bundle_id}_fish{i}.jpg"
                
                imageExists = super().checkIfImageDuplicateExists(crop)
                if not imageExists:
                    cv2.imwrite(os.path.join(self.output_single_folder, crop_name), crop)
                    self.save_images.append(crop)
                    
    def extractImageFromPrediciton(self, results, img_path, image_name):
        for i, box in enumerate(results[0].boxes):
                cls = int(box.cls)
                conf = float(box.conf)

                # Only save if confidence is decent
                if conf > 0.1:
                    img = cv2.imread(img_path)
                    
                    crop = super().getBoxImage(box, img)

                    crop_name = f"{os.path.splitext(image_name)[0]}_fish{i}.jpg"
                    
                    imageExists = super().checkIfImageDuplicateExists(crop)
                    
                    if not imageExists:
                        if crop.shape[0] > 1500:
                            #process to extract single from bundle
                            self.extractFishImageFromBundle(crop, image_name, i)
                        else:
                            #save
                            if not os.path.isfile(crop_name):
                                cv2.imwrite(os.path.join(self.output_single_folder, crop_name), crop)
                                self.save_images.append(crop)
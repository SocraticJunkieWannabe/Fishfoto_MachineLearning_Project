from ultralytics import YOLO
import cv2
import os
import numpy as np


class Sorter():
    
    def __init__(self, model,  DEBUG_FLAG : bool = False):
        self.model = model
        self.save_images = []
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
    
    def extractFishImageFromBundle(self, img, bundle_id : int, output_folder, image_name):
        results = self.model.predict(img, 
                                imgsz=1280, 
                                conf=0.3, 
                                iou=0.5,
                                max_det=150,
                                verbose=False
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

                    cv2.imwrite(f"{output_folder}/{image_name}_bundle-{bundle_id}_fish_{j}.png", rgba)
        pass
    
    def extractImageFromPrediciton(self, results, img_path, image_name, output_folder):
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
                        self.extractFishImageFromBundle(crop, i, output_folder, image_name)
                    else:
                        cv2.imwrite(f"{output_folder}/{image_name}_fish_{j}.png", super().getBoxImage(box, rgba))
        pass
    
class BoxSorter(Sorter):
    
    def extractFishImageFromBundle(self, img, image_name, bundle_id : int, output_folder):
        results = self.model.predict(img, 
                                imgsz=1280, 
                                conf=0.3, 
                                iou=0.5,
                                max_det=150,
                                verbose=False
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
                    cv2.imwrite(os.path.join(output_folder, crop_name), crop)
                    self.save_images.append(crop)
                    
    def extractImageFromPrediciton(self, results, img_path, image_name, output_folder):
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
                            self.extractFishImageFromBundle(crop, image_name, i, output_folder)
                        else:
                            #save
                            if not os.path.isfile(crop_name):
                                cv2.imwrite(os.path.join(output_folder, crop_name), crop)
                                self.save_images.append(crop)
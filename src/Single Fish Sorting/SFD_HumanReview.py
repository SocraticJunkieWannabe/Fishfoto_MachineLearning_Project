from ultralytics import YOLO
import cv2
import os
import numpy as np
import time

import Sorters  


class SingleFishClassification():
    
    def __init__(self, model_file_name, mode : str = "test", sorterType : str = "shape"):
        """
        
        Class used to split batch images of fishes into single images, stored in the "./data/Identified_Single_Fishes" folder
        
        Input:
        
        mode (str) : whether the classifaction is being tested (test), or used for production (prod)
        
        """
        
        self.mode = mode
        self.sorterType = sorterType
        
        self.data_path = "../../data/SingleFishes/"
        
        self.folder_output = {
            "test" : {"test": f"Tests/{sorterType}_fish_crops"},
            "prod" : {
                "herring" : "ISF_humanReview/herring",
                "sprat" : "ISF_humanReview/sprat"
            }
        }
        
        self.folder_input = {
            "test" : {"test" : "Tests/test_raw_images"},
            "prod" : {
                "herring" : "../Labeled Images/herring",
                "sprat" : "../Labeled Images/sprat"
            }
        }
        
        self.initOuputFolders()
        
        self.model = YOLO(model_file_name)  
    
        self.sorter = self.initSorter()
        
        # Minimum contour area in pixels to keep a fish detection
        self.min_fish_area = 500  # Adjust this value based on your images

        pass
    
    def initSorter(self):
        if self.sorterType == "shape":
            sorter = Sorters.ShapeSorter(self.model)
            return sorter
        elif self.sorterType == "box":
            sorter = Sorters.BoxSorter(self.model)
            return sorter
        else:
            print("ERROR: No sorter matching the type")   
            quit() 
            
    
    def initOuputFolders(self):
        
        if self.mode == "prod":
            os.makedirs(f"{self.data_path}/ISF_humanReview", exist_ok=True)
            
            for folder_path in self.folder_output["prod"].values():
                os.makedirs(f"{self.data_path}/{folder_path}", exist_ok=True)
                
        elif self.mode == "test":
            os.makedirs(f"{self.data_path}/Tests", exist_ok=True)
            
            os.makedirs(f"{self.data_path}/{self.folder_output[self.mode][self.mode]}", exist_ok=True)
        
        pass

    def extractFishesFromImages(self):
        # 2. Input and output folders

        self.extraction_data = {"amounts" : {},
                                "time" : 0
                                }
        
        start_time = time.time()

        for input_type in self.folder_input[self.mode].keys():
            
            input_path = f"{self.data_path}/{self.folder_input[self.mode][input_type]}"
            output_path = f"{self.data_path}/{self.folder_output[self.mode][input_type]}"
            
            # 3. Loop over all images
            itr = 0
            for image_name in os.listdir(input_path):
                itr +=1
                print(f"Treating Image Number {itr} of Class {input_type}")
                if not image_name.lower().endswith(('.jpg', '.png', '.jpeg')):
                    continue
                img_path = os.path.join(input_path, image_name)
                results = self.model(img_path,
                                iou=0.5,         # Allow more overlapping detections
                                imgsz=1280,      # Larger input image size → better small-object detection
                                max_det=300,      # Increase number of detections per image
                                verbose=False
                                )

                # 4. For each detection, crop the fish and save it
                #extractImageBoxFromPrediciton(results, img_path, image_name, output_single_folder)
                #extractImageShapeFromPrediciton(results, img_path, output_bundle_folder)
                self.sorter.extractImageFromPrediciton(results, img_path, image_name, output_path)
                
            self.extraction_data["amounts"][f"{input_type}"] = len(os.listdir(output_path))
            
        self.extraction_data["time"] = time.time() - start_time 
            
        self.imageExtractionDataPrinting()
            
        pass
    
    def imageExtractionDataPrinting(self):
        
        print(f"In {self.extraction_data['time']} sec:")
        
        for key in self.extraction_data["amounts"].keys():
            print(f"Extracted {self.extraction_data['amounts'][key]} Images from class {key}")
        
        pass
    
    def reviewDetections(self, img_path, results):
        """
        Display YOLO detections with segmentation masks (not just rectangles)
        
        Returns:
            tuple: (decision, results)
        """
        # Load original image
        img = cv2.imread(img_path)
        display_img = img.copy()
        
        # Draw segmentation masks and contours instead of boxes
        if results[0].masks is not None:
            masks = results[0].masks.data.cpu().numpy()
            boxes = results[0].boxes
            
            for idx, (mask, box) in enumerate(zip(masks, boxes)):
                # Resize mask to image size
                mask_resized = cv2.resize(mask, (img.shape[1], img.shape[0]))
                mask_binary = (mask_resized > 0.5).astype(np.uint8)
                
                # Find contours of the mask
                contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # Generate random color for this fish
                color = tuple(np.random.randint(0, 255, 3).tolist())
                
                # Draw the contour (fish outline)
                cv2.drawContours(display_img, contours, -1, color, 2)
                
                # Fill with semi-transparent color
                overlay = display_img.copy()
                cv2.drawContours(overlay, contours, -1, color, -1)
                display_img = cv2.addWeighted(display_img, 0.7, overlay, 0.3, 0)
                
                # Add label
                conf = float(box.conf[0])
                x1, y1 = map(int, box.xyxy[0][:2].cpu().numpy())
                cv2.putText(display_img, f"Fish {idx+1} ({conf:.2f})", (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        else:
            # Fallback to bounding boxes if no masks available
            display_img = results[0].plot()
        
        # Display the image
        cv2.imshow('YOLO Detection Review - Press Y(approve)/S(skip)/E(edit)/Q(quit)', display_img)
        
        while True:
            key = cv2.waitKey(0) & 0xFF
            
            if key == ord('y'):
                cv2.destroyAllWindows()
                return 'approve', results
            elif key == ord('s'):
                cv2.destroyAllWindows()
                return 'skip', results
            elif key == ord('e'):
                cv2.destroyAllWindows()
                return 'edit', results
            elif key == ord('q'):
                cv2.destroyAllWindows()
                return 'quit', results
    
    def editDetections(self, img_path, results):
        """
        Interactive editor using segmentation masks (fish-shaped boundaries)
        
        Instructions:
        - Click on a fish contour to select it
        - Click and drag to add rectangular region (will be converted to mask)
        - Press 'D' to delete selected detection
        - Press 'A' to accept changes
        - Press 'C' to cancel
        
        Returns:
            Modified detections or None if cancelled
        """
        img = cv2.imread(img_path)
        original_img = img.copy()
        
        # Extract existing masks and boxes from YOLO results
        detections = []
        if results[0].masks is not None:
            masks = results[0].masks.data.cpu().numpy()
            boxes = results[0].boxes
            
            for idx, (mask, box) in enumerate(zip(masks, boxes)):
                # Resize mask to image size
                mask_resized = cv2.resize(mask, (img.shape[1], img.shape[0]))
                mask_binary = (mask_resized > 0.5).astype(np.uint8)
                
                # Find contours
                contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                conf = float(box.conf[0].cpu().numpy())
                cls = int(box.cls[0].cpu().numpy())
                
                # Add ALL contours from this mask (not just the first one)
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area >= self.min_fish_area:  # Filter by minimum fish area
                        detections.append({
                            'contour': contour,
                            'mask': mask_binary,
                            'conf': conf,
                            'cls': cls,
                            'type': 'mask',
                            'area': area
                        })
                    else:
                        print(f"Filtered out small detection (area: {area:.0f} < {self.min_fish_area})")
        
        # Drawing state in edit mode
        drawing = False
        current_rect = None
        selected_idx = None
        
        def point_in_contour(point, contour):
            """Check if point is inside contour"""
            return cv2.pointPolygonTest(contour, point, False) >= 0
        
        def draw_all_detections(image):
            """Draw all detections with their contours"""
            display_img = image.copy()
            
            for idx, det in enumerate(detections):
                color = (0, 255, 0) if idx == selected_idx else (255, 100, 0)
                
                # Draw contour
                cv2.drawContours(display_img, [det['contour']], -1, color, 2)
                
                # Semi-transparent fill
                overlay = display_img.copy()
                cv2.drawContours(overlay, [det['contour']], -1, color, -1)
                display_img = cv2.addWeighted(display_img, 0.8, overlay, 0.2, 0)
                
                # Label
                M = cv2.moments(det['contour'])
                if M['m00'] != 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    area = det.get('area', cv2.contourArea(det['contour']))
                    label = f"Fish {idx+1} ({det['conf']:.2f}) {area:.0f}px"
                    cv2.putText(display_img, label, (cx-40, cy),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            return display_img
        
        def mouse_callback(event, x, y, flags, param):
            nonlocal drawing, current_rect, selected_idx, img
            
            if event == cv2.EVENT_LBUTTONDOWN:
                # Check if clicking inside existing contour
                for idx, det in enumerate(detections):
                    if point_in_contour((x, y), det['contour']):
                        selected_idx = idx
                        img = draw_all_detections(original_img)
                        cv2.imshow(window_name, img)
                        print(f"Selected Fish {idx+1}")
                        return
                
                # Start drawing new rectangle
                selected_idx = None
                drawing = True
                current_rect = [x, y, x, y]
            
            elif event == cv2.EVENT_MOUSEMOVE:
                if drawing:
                    current_rect[2:] = [x, y]
                    temp_img = draw_all_detections(original_img)
                    cv2.rectangle(temp_img, (current_rect[0], current_rect[1]), 
                                (current_rect[2], current_rect[3]), (0, 255, 255), 2)
                    cv2.imshow(window_name, temp_img)
            
            elif event == cv2.EVENT_LBUTTONUP:
                if drawing:
                    drawing = False
                    current_rect[2:] = [x, y]
                    
                    # Get bounding box coordinates
                    x1 = min(current_rect[0], current_rect[2])
                    y1 = min(current_rect[1], current_rect[3])
                    x2 = max(current_rect[0], current_rect[2])
                    y2 = max(current_rect[1], current_rect[3])
                    
                    # Make sure the box has a valid size
                    if abs(x2 - x1) > 10 and abs(y2 - y1) > 10:
                        print(f"Running YOLO segmentation on selected region...")
                        
                        # Crop the region from the original image
                        cropped_region = original_img[y1:y2, x1:x2].copy()
                        
                        # Run YOLO on the cropped region
                        crop_results = param['model'](cropped_region, 
                                                      iou=0.5, 
                                                      imgsz=640,
                                                      verbose=False)
                        
                        # Extract masks from the cropped region
                        if crop_results[0].masks is not None:
                            crop_masks = crop_results[0].masks.data.cpu().numpy()
                            crop_boxes = crop_results[0].boxes
                            
                            added_count = 0
                            for mask, box in zip(crop_masks, crop_boxes):
                                # Resize mask to cropped region size
                                mask_resized = cv2.resize(mask, (x2-x1, y2-y1))
                                mask_binary = (mask_resized > 0.5).astype(np.uint8)
                                
                                # Create full-size mask and offset to original position
                                full_mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
                                full_mask[y1:y2, x1:x2] = mask_binary
                                
                                # Find contours
                                contours, _ = cv2.findContours(full_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                                
                                conf = float(box.conf[0].cpu().numpy())
                                cls = int(box.cls[0].cpu().numpy())
                                
                                # Add all contours from this detection
                                for contour in contours:
                                    area = cv2.contourArea(contour)
                                    if area >= param['min_area']:
                                        detections.append({
                                            'contour': contour,
                                            'mask': full_mask,
                                            'conf': conf,
                                            'cls': cls,
                                            'type': 'manual_segmented',
                                            'area': area
                                        })
                                        added_count += 1
                                    else:
                                        print(f"Filtered out small detection (area: {area:.0f} < {param['min_area']})")
                            
                            print(f"Added {added_count} fish detection(s) from region. Total: {len(detections)}")
                        else:
                            print("No fish detected in selected region")
                    
                    current_rect = None
                    img = draw_all_detections(original_img)
                    cv2.imshow(window_name, img)
        
        window_name = 'Edit Detections - Click fish to select, Drag to add (auto-segment), D=delete, A=accept, C=cancel'
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, mouse_callback, {'model': self.model, 'min_area': self.min_fish_area})
        
        img = draw_all_detections(original_img)
        cv2.imshow(window_name, img)
        
        print("\nEdit Mode (Auto-Segmentation):")
        print("- Click on a fish outline to select it")
        print("- Click and drag to draw a box - YOLO will auto-segment fish inside it")
        print("- Press 'D' to delete selected fish")
        print("- Press 'A' to accept changes")
        print("- Press 'C' to cancel")
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('a'):  # Accept
                cv2.destroyAllWindows()
                return detections, True
            
            elif key == ord('c'):  # Cancel
                cv2.destroyAllWindows()
                return None, False
            
            elif key == ord('d'):  # Delete
                if selected_idx is not None and selected_idx < len(detections):
                    del detections[selected_idx]
                    selected_idx = None
                    img = draw_all_detections(original_img)
                    cv2.imshow(window_name, img)
                    print(f"Deleted detection. {len(detections)} remaining.")
    
    def extractFromCustomBoxes(self, detections, img_path, image_name, output_path):
        """
        Extract and save fish images using segmentation masks/contours
        """
        img = cv2.imread(img_path)
        base_name = os.path.splitext(image_name)[0]
        
        for idx, det in enumerate(detections):
            # Get bounding rectangle from contour
            x, y, w, h = cv2.boundingRect(det['contour'])
            
            # Create masked image - extract only the fish shape
            mask = det['mask']
            
            # Apply mask to get only the fish pixels
            masked_img = cv2.bitwise_and(img, img, mask=mask)
            
            # Crop to bounding box
            cropped_fish = masked_img[y:y+h, x:x+w]
            
            # Generate output filename
            output_filename = f"{base_name}_fish_{idx+1}.jpg"
            output_full_path = os.path.join(output_path, output_filename)
            
            # Save the cropped image
            cv2.imwrite(output_full_path, cropped_fish)
        
        print(f"Extracted {len(detections)} fish images from {image_name}")
    
    
    def extractFishesFromImagesWithReview(self):
        """
        Extract fishes from images with human review step
        """
        self.extraction_data = {"amounts": {}, "time": 0, "skipped": 0, "approved": 0}
        
        start_time = time.time()

        for input_type in self.folder_input[self.mode].keys():
            
            input_path = f"{self.data_path}/{self.folder_input[self.mode][input_type]}"
            output_path = f"{self.data_path}/{self.folder_output[self.mode][input_type]}"
            
            itr = 0
            for image_name in os.listdir(input_path):
                itr += 1
                print(f"Reviewing Image Number {itr} of Class {input_type}")
                
                if not image_name.lower().endswith(('.jpg', '.png', '.jpeg')):
                    continue
                    
                img_path = os.path.join(input_path, image_name)
                results = self.model(img_path,
                                iou=0.5,
                                imgsz=1280,
                                max_det=300,
                                verbose=False)
                
                # Human review step
                decision, results = self.reviewDetections(img_path, results)
                
                if decision == 'quit':
                    print("Review process terminated by user.")
                    cv2.destroyAllWindows()
                    return
                elif decision == 'skip':
                    print(f"Skipped: {image_name}")
                    self.extraction_data["skipped"] += 1
                    continue
                elif decision == 'edit':
                    # Enter edit mode
                    edited_boxes, accepted = self.editDetections(img_path, results)
                    if accepted and edited_boxes:
                        print(f"Edited and approved: {image_name}")
                        # Use edited boxes for extraction
                        self.extractFromCustomBoxes(edited_boxes, img_path, image_name, output_path)
                        self.extraction_data["approved"] += 1
                    else:
                        print(f"Edit cancelled, skipping: {image_name}")
                        self.extraction_data["skipped"] += 1
                    continue
                elif decision == 'approve':
                    print(f"Approved: {image_name}")
                    self.extraction_data["approved"] += 1
                    # Extract and save the fishes
                    self.sorter.extractImageFromPrediciton(results, img_path, image_name, output_path)
                
            self.extraction_data["amounts"][f"{input_type}"] = len(os.listdir(output_path))
            
        self.extraction_data["time"] = time.time() - start_time 
        cv2.destroyAllWindows()
        self.imageExtractionDataPrinting()
        print(f"Approved: {self.extraction_data['approved']}, Skipped: {self.extraction_data['skipped']}")
        
        pass
        
            

sorterType = "shape"

mode = "prod"

model_name = "yolo11x-seg.pt"

# Use extractFishesFromImagesWithReview() for human review mode
# Use extractFishesFromImages() for automatic mode
classifier = SingleFishClassification(model_name, mode, sorterType)
classifier.extractFishesFromImagesWithReview()  # Human review enabled


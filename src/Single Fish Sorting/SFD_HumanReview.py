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
                "herring" : "SingleFishes/Identified_Single_Fishes/herring",
                "sprat" : "SingleFishes/Identified_Single_Fishes/sprat"
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
            os.makedirs(f"{self.data_path}/Identified_Single_Fishes", exist_ok=True)
            
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
        
            

sorterType = "shape"

mode = "prod"

model_name = "yolo11x-seg.pt"

SingleFishClassification(model_name, mode, sorterType).extractFishesFromImages()


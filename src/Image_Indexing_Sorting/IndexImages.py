import json
import os
import shutil


class IndexImages():
    
    def __init__(self, stock_images_path, target_sorted_copy_path, config_file_path):
        
        self.DEBUG_FLAG = True

        self.config_file_path = config_file_path

        self.translate = {
            "räim" : "herring",
            "kilu": "sprat",
            "proov": "mixture"
        }
        
        self.stock_images_path = stock_images_path
        self.target_sorted_copy_path = target_sorted_copy_path
        
        self.standardizeImageNames()
        self.setImageConfigFile(self.extractImageNames())
        
        pass
    


    def getImageConfigFile(self):
        with open(self.config_file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        return data
        
        
    def setImageConfigFile(self, data):
        
        json_str = json.dumps(data, indent=4, ensure_ascii=False)
        with open(self.config_file_path, "w", encoding = 'utf8') as f:
            f.write(json_str)
            
        pass
    
    def standardizeImageNames(self):
        folderContent = os.listdir(self.stock_images_path)
        for fileName in folderContent:
            if fileName.split(".")[-1] == "jpg":
                old_filename = fileName
                fileName = fileName.replace("r_im", "räim")
                fileName = fileName.replace("_", " ")
                fileName = fileName.replace("-", " ")
            
                fileName = fileName.replace(" ", "_")
                
                
                os.rename(f"{self.stock_images_path}/{old_filename}", f"{self.stock_images_path}/{fileName}")
        pass

    def extractImageNames(self):
        
        config = {
        "herring" : [],
        "sprat": [],
        "mixture": []
        }
        
        folderContent = os.listdir(self.stock_images_path)
        for fileName in folderContent:
            if fileName.split(".")[-1] == "jpg":
                typeRaw = fileName.split("_")[1]
                
                typeEst = ""
                for character in list(typeRaw):
                    if not character.isdigit():
                        typeEst = typeEst + character
                
                if typeEst in self.translate.keys():
                    typeEng = self.translate[typeEst]
                    config[typeEng].append(fileName)
        
        return config
    
    def sortImages(self):
        data = self.getImageConfigFile()

        os.makedirs(f"{self.target_sorted_copy_path}", exist_ok=True)

        for key in data.keys():
            for image_name in data[key]:
                os.makedirs(f"{self.target_sorted_copy_path}/{key}", exist_ok=True)
                shutil.copyfile(f"{self.stock_images_path}/{image_name}", f"{self.target_sorted_copy_path}/{key}/{image_name}")
                
                    
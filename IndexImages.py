import json
import os


class IndexImages():
    
    def __init__(self):
        
        self.DEBUG_FLAG = True

        self.config_file_path = "indexed_images_config.json"

        self.translate = {
            "räim" : "herring",
            "kilu": "sprat",
            "proov": "mixture"
        }
        
        self.setImageConfigFile(self.extractImageNames())
        
        pass
    


    def getImageConfigFile(self):
        with open(self.config_file_path, "r") as f:
            data = json.load(f)
        
        return data
        
        
    def setImageConfigFile(self, data):
        
        json_str = json.dumps(data, indent=4, ensure_ascii=False)
        with open(self.config_file_path, "w", encoding = 'utf8') as f:
            f.write(json_str)
            
        pass

    def extractImageNames(self):
        
        config = {
        "herring" : [],
        "sprat": [],
        "mixture": []
        }
        
        folderContent = os.listdir("./Images")
        for fileName in folderContent:
            if fileName.split(".")[-1] == "jpg":
                typeRaw = fileName.split(" ")[-2]
                
                typeEst = ""
                for character in list(typeRaw):
                    if not character.isdigit():
                        typeEst = typeEst + character
                
                if typeEst in self.translate.keys():
                    typeEng = self.translate[typeEst]
                    config[typeEng].append(fileName)
        
        return config
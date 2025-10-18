import os 

def deleteFolderContent(folder_path : str):
    for file_name in os.listdir(folder_path):
        os.remove(f"{folder_path}/{file_name}")
    pass 

def cleanFishAndBundleCrops():
    
    deleteFolderContent("../../data/bundle_crops")
    deleteFolderContent("../../data/fish_crops")
    
    pass

cleanFishAndBundleCrops()
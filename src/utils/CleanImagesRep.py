import os 

def deleteFolderContent(folder_path : str):
    for file_name in os.listdir(folder_path):
        os.remove(f"{folder_path}/{file_name}")
    pass 

def cleanFishAndBundleCrops():
    
    folders = ["test_bundle_crops", "shape_fish_crops", "box_fish_crops", "pseudo_mixtures"]
    
    for folder in folders:
        deleteFolderContent(f"../../data/Tests/{folder}")
    
    
    pass

cleanFishAndBundleCrops()
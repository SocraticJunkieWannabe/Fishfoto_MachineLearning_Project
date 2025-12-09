import os 

"""
===============================================================================
 Description   :
    This script provides utility functions to clean specific folders within
    the data repository by deleting all files inside them. It is mainly used
    to reset testing and production directories before running new experiments
    or generating new data crops.

 Directory Structure  :
    data/
        ├── Tests/
        │      ├── test_bundle_crops/
        │      ├── shape_fish_crops/
        │      ├── box_fish_crops/
        │      └── pseudo_mixtures/
        └── Identified_Single_Fishes/
               ├── herring/
               └── sprat/

 Functions     :
    deleteFolderContent(folder_path: str)
        Deletes all files inside the specified folder (non-recursive).

    cleanTestFishAndBundleCrops()
        Empties crop-related test folders under the 'Tests' directory.

    cleanProdCrops()
        Empties the fish category folders ('herring' and 'sprat') in the
        'Identified_Single_Fishes' directory.

    cleanAllReps()
        Executes all cleanup routines (test and production folders).

 Notes :
    - This script only deletes files, not subdirectories.
    - Ensure you have backups if the data is valuable.
    - Folder paths are relative to the script’s location.
===============================================================================
"""


def deleteFolderContent(folder_path : str):
    for file_name in os.listdir(folder_path):
        os.remove(f"{folder_path}/{file_name}")
    pass 

def cleanTestFishAndBundleCrops():
    
    folders = ["test_bundle_crops", "shape_fish_crops", "box_fish_crops", "pseudo_mixtures"]
    
    for folder in folders:
        deleteFolderContent(f"../../data/Tests/{folder}")
    
    
    pass

def cleanProdCrops():
    folders = ["herring", "sprat"]
    
    for folder in folders:
        deleteFolderContent(f"../../data/Identified_Single_Fishes/{folder}")
    
    pass

def cleanLabeledImages():
    folders = ["herring", "sprat","mixture"]
    
    for folder in folders:
        deleteFolderContent(f"../../data/Labeled Images/{folder}")
    
    pass

def cleanRealMixImages():
    
    folders = ["training","testing"]
    
    for folder in folders:
        deleteFolderContent(f"../../data/Real_Mixtures_Augmented/{folder}")
    
    pass

def cleanAllReps():
    cleanTestFishAndBundleCrops()
    cleanProdCrops()
    pass

#cleanAllReps()
#cleanLabeledImages()
cleanRealMixImages()
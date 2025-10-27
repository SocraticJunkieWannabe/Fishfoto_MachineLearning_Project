from PIL import Image
import glob

"""
===============================================================================
 Description   : 
    This script converts all PNG images located in the 'Pseudo_Mixtures' 
    directory into JPEG format and saves them in the 'Converted_Mixtures' 
    directory. It uses the Python Imaging Library (PIL) to handle image 
    conversion and removes alpha channels if present.

 Directory Structure  :
    data/
        ├── Pseudo_Mixtures/
        │      ├── image_1.png
        │      ├── image_2.png
        │      └── ...
        └── Tests/
               └── Converted_Mixtures/

 Functions     :
    convertImage(path, save_path)
        Converts a single PNG image to JPG and saves it to the target directory.

    convertRepository(paths: list, save_path)
        Iterates over a list of image paths and converts each one to JPG format.

 Notes :
    - The save directory must exist before running this script.
    - Ensures RGB conversion to avoid alpha channel issues.

===============================================================================
"""

data_path = "../../data"
pseudo_mixtures_paths = glob.glob(f"{data_path}/Pseudo_Mixtures/*.png") 
save_path = f"{data_path}/Tests/Converted_Mixtures"

def convertImage(path, save_path):
    
    # Open the PNG image
    png_image = Image.open(path)

    img_name = path.split("\\")[-1].split(".png")[-2]

    # Convert to RGB (to remove alpha channel if present)
    rgb_image = png_image.convert("RGB")

    # Save as JPG
    rgb_image.save(f"{save_path}/{img_name}.jpg", "JPEG")
    
    pass

def convertRepository(paths : list, save_path):
    
    for path in paths:
        convertImage(path, save_path)
    
    pass

convertRepository(pseudo_mixtures_paths, save_path)
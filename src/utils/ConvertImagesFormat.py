from PIL import Image
import glob


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
from PIL import Image
import random, glob
import os
import time

data_path = "../../data"

real_mixture_path = f"{data_path}/Labeled Images/mixture"
augmented_mixtures_path = f"{data_path}/Augmented_Real_Mixtures"
mixture_paths = glob.glob(f"{real_mixture_path}/*.png") 

os.makedirs(f"{augmented_mixtures_path}", exist_ok=True)
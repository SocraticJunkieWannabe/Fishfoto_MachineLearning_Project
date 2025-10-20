from PIL import Image
import random, glob
import os
import time

data_path = "../../data"

pseudo_mixture_path = f"{data_path}/Pseudo_Mixtures"
single_herring_path = f"{data_path}/Identified_Single_Fishes/herring"
single_sprat_path = f"{data_path}/Identified_Single_Fishes/sprat"

herring_paths = glob.glob(f"{single_herring_path}/*.png") 
sprat_paths = glob.glob(f"{single_sprat_path}/*.png") 

def createArrayOfSingleFish(spratRatio, total_images):
    
    n_sprat = int(total_images * spratRatio)
    n_herring = total_images - n_sprat
    
    selected_sprat = random.choices(sprat_paths, k=n_sprat)
    selected_herring = random.choices(herring_paths, k=n_herring)
    
    selected_all = selected_sprat + selected_herring
    random.shuffle(selected_all)
    
    return selected_all

def createPseudoMixturesDataset(fish_per_layer : int, num_layers : int = 2, sizeDataset : int = 1000, spratRatioBoundaries : list = [0.01, 0.99]):
    os.makedirs(pseudo_mixture_path, exist_ok=True)
    
    total_images = fish_per_layer * num_layers
    
    for i in range(sizeDataset):
        
        start_time = time.time()
        
        spratRatio = random.uniform(spratRatioBoundaries[0], spratRatioBoundaries[1])
        image = createPseudoMixtureImage(createArrayOfSingleFish(spratRatio, total_images), num_layers, fish_per_layer)
        image.save(f"{pseudo_mixture_path}/Image_{i+1}_sprat-ratio_{round(spratRatio, 3)}.png")
        
        print(f"Image {i+1} created, taking {round(time.time() - start_time)} seconds")
    
    pass

def createPseudoMixtureImage(fish_paths : str, num_layers : int, fish_per_layer : int):
    
    # Load all fish image paths
    canvas_size = (4600, 3500)

    bundle = Image.new("RGBA", canvas_size, (0,0,0,0))

    index = 0

    for _ in range(num_layers):
        layer = Image.new("RGBA", canvas_size, (0,0,0,0))
        for _ in range(fish_per_layer):
            fish = Image.open(fish_paths[index]).convert("RGBA")

            # Random resize & rotation
            scale = random.uniform(0.5, 1.5)
            fish = fish.resize((int(fish.width*scale), int(fish.height*scale)))
            fish = fish.rotate(random.randint(0, 360), expand=True)

            # Allow partial cropping — fish can go outside frame
            x = random.randint(-fish.width//2, canvas_size[0] - fish.width//2)
            y = random.randint(-fish.height//2, canvas_size[1] - fish.height//2)

            layer.paste(fish, (x, y), fish)
            
            index += 1

        bundle = Image.alpha_composite(bundle, layer)
        
    # Ensure final image fits canvas exactly (cropping overflow)
    bundle = bundle.crop((0, 0, canvas_size[0], canvas_size[1]))
    
    return bundle

createPseudoMixturesDataset(300, 2, 1000, [0.01, 0.99])



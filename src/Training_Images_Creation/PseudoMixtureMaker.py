from PIL import Image
import random, glob
import os
import time
import numpy as np
import pandas as pd

data_path = "../../data"

pseudo_mixture_path = f"{data_path}/Pseudo_Mixtures"
single_herring_path = f"{data_path}/SingleFishes/Identified_Single_Fishes/herring"
single_sprat_path = f"{data_path}/SingleFishes/Identified_Single_Fishes/sprat"
single_stickleback_path = f"{data_path}/SingleFishes/Identified_Single_Fishes/stickleback"
single_smelt_path = f"{data_path}/SingleFishes/Identified_Single_Fishes/smelt"

herring_paths = glob.glob(f"{single_herring_path}/*.png") 
sprat_paths = glob.glob(f"{single_sprat_path}/*.png") 
stickleback_paths = glob.glob(f"{single_stickleback_path}/*.png") 
smelt_paths = glob.glob(f"{single_smelt_path}/*.png") 

def loadRealPercentagesDF():
    df = pd.read_csv('../../data/percentages.csv', sep=";")
    df = df.fillna(0)

    # Select the two rows
    stick = df[df['Haul nr. / Species percentage'] == 'Stickleback']
    nine  = df[df['Haul nr. / Species percentage'] == '9-spine stickleback']

    # Convert numeric columns
    num_cols = df.columns[1:]
    stick[num_cols] = stick[num_cols].apply(pd.to_numeric, errors='coerce')
    nine[num_cols]  = nine[num_cols].apply(pd.to_numeric, errors='coerce')

    # Sum the two rows
    combined = stick[num_cols].fillna(0).values + nine[num_cols].fillna(0).values

    # Build new row
    new_row = pd.DataFrame(
        [['Stickleback'] + combined.flatten().tolist()],
        columns=df.columns
    )

    # New dataframe with rows except the original two
    df_new = pd.concat([
        df[~df['Haul nr. / Species percentage'].isin(['Stickleback',
                                                    '9-spine stickleback'])],
        new_row
    ], ignore_index=True)

    df = df_new
    
    return df

def createArrayOfSingleFish(ratios, total_images):
    
    counts = [int(total_images * r) for r in ratios]
    diff = total_images - sum(counts)  # missing/extra due to truncation

    # Adjust counts randomly to make sum exactly total_images
    for _ in range(diff):
        counts[random.randint(0, 3)] += 1  # add 1 to random species

    n_sprat, n_herring, n_smelt, n_stickleback = counts
    
    selected_sprat = random.choices(sprat_paths, k=n_sprat)
    selected_herring = random.choices(herring_paths, k=n_herring)
    selected_stickleback = random.choices(stickleback_paths, k=n_stickleback)
    selected_smelt = random.choices(smelt_paths, k=n_smelt)
    
    selected_all = selected_sprat + selected_herring + selected_stickleback + selected_smelt
    random.shuffle(selected_all)
    
    return selected_all

def createPseudoMixturesDataset(fish_per_layer : int, num_layers : int = 2, sizeDataset : int = 1000, spratRatioBoundaries : list = [0.01, 0.99]):
    os.makedirs(pseudo_mixture_path, exist_ok=True)
    
    total_images = fish_per_layer * num_layers
    
    images_already_generated = len(os.listdir(pseudo_mixture_path))
    
    df = loadRealPercentagesDF()
    
    row_average = df.iloc[:, 1:].sum(axis=1)/len(df.columns)
    ratioConcentraion = row_average.to_list()
    
    ratiosDf = pd.DataFrame(columns=["path", "spratRatio", "herringRatio", "smeltRatio", "sticklebackRatio"])
    
    for i in range(sizeDataset - images_already_generated):
        
        start_time = time.time()
        
        ratios = np.random.dirichlet(ratioConcentraion, size=1)[0]
        image = createPseudoMixtureImage(createArrayOfSingleFish(ratios, total_images), num_layers, fish_per_layer)
        
        image = image.convert("RGB")
        path = f"{pseudo_mixture_path}/Image_{images_already_generated+i+1}.jpg"
        image.save(path, "JPEG")
    
        new_row = {"path": path, 
                   "spratRatio": round(ratios[0], 4), 
                   "herringRatio": round(ratios[1], 4), 
                   "smeltRatio": round(ratios[2], 4), 
                   "sticklebackRatio": round(ratios[3], 4)
        }
        ratiosDf = pd.concat([ratiosDf, pd.DataFrame([new_row])], ignore_index=True)
        
        print(f"Image {images_already_generated+i+1} created, taking {round(time.time() - start_time)} seconds")
        
    ratiosDf.to_csv(f"{pseudo_mixture_path}/ratios.csv", index=False)
    
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

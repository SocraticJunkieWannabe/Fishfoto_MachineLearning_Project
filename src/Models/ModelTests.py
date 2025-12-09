import PredictFromAugmentedRealMixtures.ModelLoader as aug
import PredictFromPseudoMixtures.ModelLoader as pseudo
import PredictFishTypeFromSingle.ModelLoader as single

import pandas as pd
import os
import numpy as np

from PIL import Image

import torch
from torchvision import models
from torch import nn
from PIL import Image
from torchvision import transforms

class Dataset():
    def __init__(self):
        
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

        self.df = df_new
        
        pass



class Model():
    def __init__(self):
        
        pseudoFile = "pseudo_model.pth" 
        singleFile = "single_model.pth"
        augmentedFile = "real_augemented_model.pth"

        self.device, self.model, self.transform = pseudo.loadModel(f"ModelFiles/{pseudoFile}")

        pass

class MixtureTest():
    def __init__(self, model : Model, df):
        self.tested_model = model
        self.df = df
        pass

    def predict_ratio(self, model, image_path):
        image = Image.open(image_path).convert("RGB")
        image = self.tested_model.transform(image).unsqueeze(0).to(self.tested_model.device)
        with torch.no_grad():
            ratio = model(image).item()
        return ratio

    def testAgainstAllStockMixtures(self, verbose : bool = False):
        mixtures_images_path = "../../data/Labeled Images/mixture"
        
        folderContent = os.listdir(mixtures_images_path)
        ratios = []
        
        for fileName in folderContent:
            print(f"testing model on {fileName}")
            output = self.testModelOnImage(f"{mixtures_images_path}/{fileName}", verbose)
            if output != None:
                ratios.append(output)
        
        if verbose:
            print(self.computeError(ratios))
        
        pass


    def testModelOnImage(self, image_path, verbose : None = False):
        
        #True Ratio Extraction
        image_file_name = image_path.split("/")[-1]
        haulNumber = image_file_name.split("_")[0].split("T")[1]
        
        if haulNumber not in self.df.columns:
            return None

        predicted_ratio = self.predict_ratio(self.tested_model.model, image_path)

        ratio_from_df = str(self.df.loc[1, haulNumber]).split(".")[0]
        actual_ratio = float(f"0.{ratio_from_df}")

        if verbose:
            #Visual Output
            print(f"Predicted ratio of sprat: {predicted_ratio:.2f}")
            print(f"Actual ratio is {actual_ratio}")
        
        return predicted_ratio, actual_ratio

    def computeError(self, data):
        #MSE
        error = 0
        for item in data:
            error += np.sqrt(np.square(item[0]-item[1]))
            
        error = error/len(data)
        
        return error
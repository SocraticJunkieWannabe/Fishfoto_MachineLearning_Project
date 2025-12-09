import PredictFromAugmentedRealMixtures.ModelLoader as aug
import PredictFromPseudoMixtures.ModelLoader as pseudo
import PredictFishTypeFromSingle.ModelLoader as single

import pandas as pd
import os
import numpy as np
import random as rand

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
    def __init__(self, _type : str):
        
        pseudoFile = "pseudo_model.pth" 
        singleFile = "single_model.pth"
        augmentedFile = "real_augemented_model.pth"
        
        typeList = ["pseudo", "single", "augmented"]
        
        if _type in typeList:
            if _type == "pseudo":
                self.device, self.model, self.transform = pseudo.loadModel(f"ModelFiles/{pseudoFile}")
            elif _type == "single":
                self.device, self.model, self.transform = pseudo.loadModel(f"ModelFiles/{singleFile}")
            elif _type == "augmented":
                self.device, self.model, self.transform = pseudo.loadModel(f"ModelFiles/{augmentedFile}")
                
                
            return
        
        print("WRONG TYPE INPUTTED")   
        
class MixtureTest():
    def __init__(self, model : Model, df, verbose : bool = True):
        self.tested_model = model
        self.df = df
        self.verbose = verbose
        pass

    def predict_ratio(self, model, image_path):
        image = Image.open(image_path).convert("RGB")
        image = self.tested_model.transform(image).unsqueeze(0).to(self.tested_model.device)
        with torch.no_grad():
            ratio = model(image).item()
        return ratio

    def testAgainstAllStockMixtures(self):
        mixtures_images_path = "../../data/Labeled Images/mixture"
        
        folderContent = os.listdir(mixtures_images_path)
        modelErrors = []
        dummyRandomErrors = []
        dummyConstantErrors = []
        
        for fileName in folderContent:
            
            output = self.testModelOnImage(f"{mixtures_images_path}/{fileName}")
            
            if output != None:
                error = self.computeError(output)
                modelErrors.append(error)
                
                if self.verbose:
                    print(f"Tested on {fileName} with {round(error, 2)} error")
                    
                #COMPUTE DUMMY ERRORS
                
                randOutput = (rand.random(), output[1])
                cstOutput = (0.5, output[1])
                
                dummyRandError = self.computeError(randOutput)
                dummyRandomErrors.append(dummyRandError)
                
                dummyCstError = self.computeError(cstOutput)
                dummyConstantErrors.append(dummyCstError)
               
        dummyRandTotal = self.computeTotalError(dummyRandomErrors)
        dummyConstTotal = self.computeTotalError(dummyConstantErrors)
        modelTotal = self.computeTotalError(modelErrors)
        
        return modelTotal, max(modelErrors), min(modelErrors), dummyRandTotal, dummyConstTotal


    def testModelOnImage(self, image_path):
        
        #True Ratio Extraction
        image_file_name = image_path.split("/")[-1]
        haulNumber = image_file_name.split("_")[0].split("T")[1]
        
        if haulNumber not in self.df.columns:
            return None

        predicted_ratio = self.predict_ratio(self.tested_model.model, image_path)

        ratio_from_df = str(self.df.loc[1, haulNumber]).split(".")[0]
        actual_ratio = float(f"0.{ratio_from_df}")

        if self.verbose:
            #Visual Output
            print(f"Predicted ratio of sprat: {predicted_ratio:.2f}")
            print(f"Actual ratio is {actual_ratio}")
        
        return (predicted_ratio, actual_ratio)

    def computeError(self, data):
        #MSE
        error = abs(data[0]-data[1])
        
        return error
    
    def computeTotalError(self, errors):
        totalError = sum(errors)/len(errors)
        return totalError
    

def testMixtures(type : str = "pseudo", verbose : bool = True):

    if type == "pseudo":
        modelName = "Pseudo"
    elif type == "augmented":
        modelName = "Real Augmented"
    
    if verbose:
        print(f"Started testing the {modelName} Mixture Model")
        
    totalError, minError, maxError, dummyRandError, dummyConstantError = MixtureTest(Model(type), Dataset().df, verbose).testAgainstAllStockMixtures()
    
    print("---------------------------------------------------------------------------------")
    
    print(f"{modelName} Mixture Model Error came back to {round(totalError, 3)}, with a range of [{round(minError, 3)}, {round(maxError, 3)}]")
    print(f"By comparaison, a random predcition gives {round(dummyRandError, 3)}, and guessing 0.5 every time gives {dummyConstantError}")
    
    if verbose:
        if totalError > dummyRandError:
            print("We have a problem, randomly guessing does better than the model")
        elif totalError > dummyConstantError:
            print("We have a problem, guessing the same number each time does better model")
    pass

testMixtures("augmented")
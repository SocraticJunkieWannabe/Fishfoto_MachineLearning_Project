# FIshfoto\_MachineLearning\_Project

/!\ Les photos sont pas dans le GitHub faut les télécharger et les mettre dans un dossier root de la rep called "Images"

(c'est les photos downloaded from le dossier "BIAS kalapildid" dans le sharepoint du pelo ou depuis ce lien: https://ibb.co/album/TBq199)
(link to the Pseudo Mixtures: https://ibb.co/album/s5Z6yQ)
THe pseudo mxitures image go into the "data/Peudo_Mixtures"

Pour bien etre a jour:

 - Importer les images de "BIAS kalapildid" dans data/Stock Images
 - run le setup.py

HOW TO DOWNLOAD IMAGES FROM ibb LINK:

    1. In the albmu Go to Embed Codes and select HTML Image from the dropdown

    2. Copy the entire text field, paste it into a text document. Save the document as album.html

    3. Open the document. Ctrl + I

    4. On the Media tab click Select All and Save As

    5. Select/ create a folder and wait for the download to complete (may take some time)

INFO:

Le fichier indexed_images_config.json contient l'indexaction des photos en fonction de leur type (poisson mix vs indetifier), créer avec IndexImages.py

Le fichier SingleFishDetection dans Single Fish Sorting permet de split les images du pelo into des images singuliers de poissons. Faut avoir fait tourner le fichierqui les index bien entendu

TO DO:

 - Fix function Image_Processing function to remove bright red and blue hues in images
 - Create feature to identify and extract just the fish mixture rectanle from the Mixtures images


 PATH TO IMPROVE:

  look into small scale jattering augmentation ml onto the test set given to only train on it
  are the fishes by themselves categorizable by a model ?  

  Instance problem -> segmetation problem


  try the approach by area, by creating pseudo mixtures with a mask (like 1 and 2s pixels) from which we can deduce area coverage and thus percentage by 
  counting the pixels
  Try to predict the mask of 1 and 2 on the test set to try go get percentages

The main issue we face in this proejct is that to get a good ratio you need to count and to count that means segmentation problem where we have to identify individual fishes
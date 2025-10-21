# FIshfoto\_MachineLearning\_Project

/!\ Les photos sont pas dans le GitHub faut les télécharger et les mettre dans un dossier root de la rep called "Images"

(c'est les photos downloaded from le dossier "BIAS kalapildid" dans le sharepoint du pelo ou depuis ce lien: https://ibb.co/album/TBq199)
(link to the Pseudo Mixtures: https://ibb.co/album/s5Z6yQ)

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
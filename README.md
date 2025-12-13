# Fishfoto Machine Learning Project

This project focuses on fish image processing and classification using machine learning techniques.

---

## Image Data (Required)

⚠️ Images are NOT included in this GitHub repository.
You must download and place them manually.

### Image Sources

- Stock Images (BIAS kalapildid)
  From this link:
  https://ibb.co/album/TBq199

- Pseudo Mixtures
  Download from:
  https://ibb.co/album/hFVwJ2

---

## Required Directory Structure

After downloading, organize the images as follows:

project-root/
├── data/
│   ├── Stock Images/        
│   └── Pseudo_Mixtures/    
├── setup.py
└── ...

---

## Setup Instructions

To ensure the project is correctly configured:

1. Import all images into:
   data/Stock Images

2. Place pseudo mixture images into:
   data/Pseudo_Mixtures

3. Run the setup script:
   python3 setup.py

---

## How to Download Images from an ibb.co Album

1. Open the album link.
2. Click "Embed Codes" and select "HTML Image" from the dropdown.
3. Copy the entire text and paste it into a text file.
4. Save the file as:
   album.html
5. Open the file in Firefox.
6. Press Ctrl + I (Page Info).
7. Go to the Media tab.
8. Click Select All → Save As.
9. Choose or create a folder and wait for the download to complete
   (this may take some time).

   If this does not work refer to the album.html file in the src folder

## Directories

The "Models" directory contains all of the model trained during the project, along with a testing file. This entails:

1. A Model to destinguish single sprats from herrings
2. A Model trained on pseudo mixtures
3. A model trained on augmented real mixtures

The "Single Fish Sorting" directory has all the files pertaining to the identfication of single fishes

The "Image Indexing Sorting" directory is for the files related to the labellling of the dataset

The "Image_Processing" directory is for the files related to processing images

Finally, the "utils" directory is for any files which proved useful to save time during devlopment 
TO DO:

 - Fix function Image_Processing function to remove bright red and blue hues in images


 PATH TO IMPROVE:

  look into small scale jattering augmentation ml onto the test set given to only train on it
  are the fishes by themselves categorizable by a model ?  

  Instance problem -> segmetation problem


  try the approach by area, by creating pseudo mixtures with a mask (like 1 and 2s pixels) from which we can deduce area coverage and thus percentage by 
  counting the pixels
  Try to predict the mask of 1 and 2 on the test set to try go get percentages

The main issue we face in this proejct is that to get a good ratio you need to count and to count that means segmentation problem where we have to identify individual fishes
import Image_Indexing_Sorting.IndexImages as II
import Training_Images_Creation.RealMixtureAugmentor

stock_images_path = "../data/Stock Images"
target_sorted_copy_path = "../data/Labeled Images"
config_file_path = "Image_Indexing_Sorting/indexed_images_config.json"

II.IndexImages(stock_images_path, target_sorted_copy_path, config_file_path).sortImages()




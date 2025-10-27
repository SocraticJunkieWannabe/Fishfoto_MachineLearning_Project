import cv2
import numpy as np


class ColorEditing():
    def __init__(self, image_path):
        
        self.img = cv2.imread(image_path)
        # Convert to HSV color space
        self.hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
        
        pass
    
    def removeStrongChannelsRGB(self, saveImageFlag : bool = False):
        
        # Define the red color range in HSV
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])

        # Create masks for red
        mask1 = cv2.inRange(self.hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(self.hsv, lower_red2, upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)

        # Invert mask to keep non-red parts
        mask_inv = cv2.bitwise_not(mask)

        # Apply the inverted mask to keep only non-red regions
        result = cv2.bitwise_and(self.img, self.img, mask=mask_inv)

        # Save or show result
        if saveImageFlag:
            cv2.imwrite("output.png", result)
            cv2.imshow("Result", result)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return result

ColorEditing("test.png").removeStrongChannelsRGB(True)










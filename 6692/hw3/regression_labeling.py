"""
Python script for labeling regression data with point and click. Run this script on
your local machine. This will not work on Jetson Nano in headless mode (SSH).

USAGE INSTRUCTIONS:

1. Transfer unlabeled images with scp from Jetson Nano to UNLABLED_DATA_PATH (local).

2. Change NUM_REGRESSION_CLASSES to the number of regression classes of your regression task.

3. Run the script. You should see a window pop up with an image to label. 

4. Inside that window, click the regression points in order of their definition. For
   example, if your regression classes are left ear, right ear, chin, then you should 
   in that order click the left ear in the image, the right ear in the image, and then
   the chin in the image. If you deviate from the order, then your labels will be 
   mismatched and your model will not train well. 
   
5. After you have clicked all the regression points in order, hit any key to move 
   to the next image. Do not click more times than there are regression points. The script
   terminates when all images in UNLABLED_DATA_PATH have been labeled.

6. The regression point coordinates are written to the text file "regression_labels.txt".
   Transfer regression_labels.txt back to the Jetson Nano to continue the lab.

"""

import cv2
import os

COLOR = (0, 255, 0) # color of point and click 
REGRESSION_LABELS_FILENAME = 'regression_labels.txt' 
UNLABLED_DATA_PATH = './data/regression/unlabeled'
NUM_REGRESSION_CLASSES = 2

HEADER = 'filename '

def click_event(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        
        print("x: ", x, ' ', "y: ", y)
        
        with open(REGRESSION_LABELS_FILENAME, 'a') as labels:
            labels.write(str(x) + ' ' + str(y) + ' ')
            
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.circle(img, (x, y), 4, COLOR, 2)
        cv2.imshow('img', img)
        
        
if __name__ == "__main__":
    
    image_names = os.listdir(UNLABLED_DATA_PATH)
    
    
    for i in range(1, NUM_REGRESSION_CLASSES+1):
        HEADER += 'CAT{}_x CAT{}_y '.format(str(i), str(i))
    HEADER += '\n'
    
    with open(REGRESSION_LABELS_FILENAME, 'w') as labels:
        
        labels.write(HEADER)
    
    for image_name in image_names:
        
        # make sure img is .jpg
        if '.jpg' in image_name:

            with open(REGRESSION_LABELS_FILENAME, 'a') as labels:
                labels.write(image_name + ' ')

            img = cv2.imread(os.path.join(UNLABLED_DATA_PATH, image_name), 1)

            cv2.imshow('img', img)

            # setting mouse handler for the image
            # and calling the click_event() function
            cv2.setMouseCallback('img', click_event)

            # wait for a key to be pressed to exit
            cv2.waitKey(0)

            # close the window
            cv2.destroyAllWindows()

            with open(REGRESSION_LABELS_FILENAME, 'a') as labels:
                labels.write('\n')
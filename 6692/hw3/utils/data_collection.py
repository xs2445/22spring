"""
This file contains functions for data collection through a Logitech webcam in a
Jupyter Notebook.

E6692 Spring 2022

YOU DO NOT NEED TO MODIFY THESE FUNCTIONS TO COMPLETE THE LAB
"""

import cv2
from IPython.display import display, clear_output
from PIL import Image
import uuid
import os

DATA_SHAPE = (256, 256)

def collect_categorical_data(class_names, dataset):
    """
    Starts the webcam data collection script to populate the specified
    image classification dataset.

    params:
        class_names (list of strings): names of the classes
        dataset (utils.datasets.ClassificationDataset): dataset object

    USAGE INSTRUCTIONS:

    1. Run collect_categorical_data() in a Jupyter cell. You should see the webcam
       video feed pop up in the cell output. Make sure you included the camera
       device when mounting the docker.
    2. This script takes a snapshot of the camera output and then gives you
       the option to save the frame and label it as the current class, discard
       the frame and take another one, continue to the next class, or exit the
       data collection script.
    3. The script will iterate through the classes defined in class_names. Each
       time you input 'c', you continue to the net class (without saving the
       current image).
    """
    cam = cv2.VideoCapture(0) # start the video feed

    try:
        iter_class = iter(class_names) # define class name iterator

        try:
            while True:

                class_name = next(iter_class) # get the next class
                print("Collecting data for class '{}' in dataset '{}'.".format(class_name, dataset.dataset_name))

                while True:

                    success, frame_array = cam.read() # read frame from camera feed

                    if not success:
                        raise Exception("Camera initialization failed or was not released from previous stream. Restart kernel.")


                    # crop to square and resize
                    frame_array = frame_array[:, 280:1000, :]
                    frame_array = cv2.resize(frame_array, DATA_SHAPE)
                    # bgr --> rgb
                    frame_array = cv2.cvtColor(frame_array, cv2.COLOR_BGR2RGB)
                    # create PIL image (what the display function expects)
                    frame = Image.fromarray(frame_array)

                    display(frame)
                    # get user input
                    val = input("Save image as '{}'? (y/n) or continue to next class? (c) or exit? (x)".format(class_name))
                    # clear current frame
                    clear_output(wait=True)
                    print(flush=True)

                    if val == 'y': # save image
                        frame_array = cv2.cvtColor(frame_array, cv2.COLOR_BGR2RGB)
                        dataset.save_image(frame_array, class_name)

                    elif val == 'c' or val == 'x':
                        break # continue to next class or exit

                if val == 'x':
                    break

            cam.release() # release camera
            print("Data collection finished.")

        except StopIteration:
            cam.release()
            print("Data collection finished.")

    except KeyboardInterrupt:
        print("Data collection finished.")
        cam.release()


def collect_regression_data(save_path):
    """
    Starts the webcam data collection script to populate the specified
    directory with unlabeled images for subsequent regression labeling.

    params:
        save_path (string): path to directory for saving the images

    USAGE INSTRUCTIONS:

    1. Run collect_regression_data() in a Jupyter cell. You should see the webcam
       video feed pop up in the cell output. Make sure you included the camera
       device when mounting the docker.
    2. This script takes a snapshot of the camera output and then gives you
       the option to save the frame. There is no labeling from this script.
       Regression labels are generated on your local machine with
       regression_labeling.py using point and click.
    3. The script will ask you if you want to save (y), discard (n), or exit (x).
       Entering (y) will save the current snapshot to save_path.
    """
    cam = cv2.VideoCapture(0) # start the video feed

    try:

        while True:

            print("Collecting unlabeled data for regression.")

            while True:

                success, frame_array = cam.read() # capture snapshot

                if not success:
                    raise Exception("Camera initialization failed or was not released from previous stream. Restart kernel.")

                # crop to square and resize
                frame_array = frame_array[:, 280:1000, :]
                frame_array = cv2.resize(frame_array, DATA_SHAPE)
                # bgr --> rgb
                frame_array = cv2.cvtColor(frame_array, cv2.COLOR_BGR2RGB)
                # convert to PIL image
                frame = Image.fromarray(frame_array)
                # display snapshot in jupyter output
                display(frame)
                # get user input
                val = input("Save image? (y/n) or exit? (x)")
                # clear snapshot
                clear_output(wait=True)
                print(flush=True)

                if val == 'y': # save image to save_path
                    frame.save(os.path.join(save_path, str(uuid.uuid1()) + '.jpg'))

                elif val == 'c' or val == 'x':
                    break

            if val == 'x':
                break

        cam.release() # release the video feed
        print("Data collection finished.")

    except KeyboardInterrupt:
        clear_output(wait=False)
        print("Data collection finished.")
        cam.release()

"""
This file contains functions for doing live inference with the Logitech webcam.

E6692 Spring 2022
"""

import torch
import torchvision.transforms as T
import cv2
import io
from PIL import Image
import IPython

from .data_collection import DATA_SHAPE

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 1
COLOR = (255, 0, 0)
THICKNESS = 2
COORD = (50, 50)
RADIUS = 5

def show_array(a, fmt='jpeg'):
    """
    Display array in Jupyter cell output using ipython widget.

    params:
        a (np.array): the input array
        fmt='jpeg' (string): the extension type for saving. Performance varies
                             when saving with different extension types.
    """
    f = io.BytesIO() # get byte stream
    Image.fromarray(a).save(f, fmt) # save array to byte stream
    display(IPython.display.Image(data=f.getvalue())) # display saved array


def live_classification(trained_weights_path, model, device, class_names):
    """
    Outputs a video feed with live inference to the jupyter cell output. This
    function is intended for classification model inference and does inference
    on GPU.

    params:
        trained_weights_path (string): path to saved model weights
        model (nn.Module): PyTorch classification model
        device (torch.device()): GPU specification
        class_names (list of strings): list of class names
    """
    print("Loading model.")
    model.load_state_dict(torch.load(trained_weights_path)) # load trained model
    model = model.eval()

    cam = cv2.VideoCapture(0) # define camera stream

    try:
        print("Video feed started.")

        while True:
            _, frame = cam.read() # read frame from video stream
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # convert raw frame from BGR to RGB

            frame = frame[:, 280:1000, :] # crop and resize
            frame = cv2.resize(frame, DATA_SHAPE)

            #####################################################################################
            # --------------------------- YOUR IMPLEMENTATION HERE ---------------------------- #
            #####################################################################################

            raise Exception('utils.live_inference.live_classification() not implemented!') # delete me

            # convert frame to tensor

            # normalize the tensor

            # transfer the frame tensor to the GPU

            # get model output from GPU -> CPU and convert to array

            # get the predicted class and class_name

            #####################################################################################
            # --------------------------- END YOUR IMPLEMENTATION ----------------------------- #
            #####################################################################################

            # draw class name on frame
            frame = cv2.putText(frame, class_name, COORD, FONT,
                       FONT_SCALE, COLOR, THICKNESS, cv2.LINE_AA)

            show_array(frame) # display the frame in JupyterLab

            IPython.display.clear_output(wait=True) # clear the previous frame

    except KeyboardInterrupt: # if interrupted
        print("Video feed stopped.")
        cam.release() # release the camera feed


def live_regression(trained_weights_path, model, device, regression_class_names):
    """
    Outputs a video feed with live inference to the jupyter cell output. This
    function is intended for regression model inference and does inference
    on GPU.

    params:
        trained_weights_path (string): path to saved model weights
        model (nn.Module): PyTorch classification model
        device (torch.device()): GPU specification
        regression_class_names (list of strings): list of regression class names
    """
    print("Loading model.")
    model.load_state_dict(torch.load(trained_weights_path)) # load trained model
    model = model.eval()

    cam = cv2.VideoCapture(0) # define camera stream

    try: # start video feed
        print("Video feed started.")

        while True:
            _, frame = cam.read() # read frame from video stream

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # convert raw frame from BGR to RGB
            # crop and resize frame
            frame = frame[:, 280:1000, :]
            frame = cv2.resize(frame, DATA_SHAPE)

            #####################################################################################
            # --------------------------- YOUR IMPLEMENTATION HERE ---------------------------- #
            #####################################################################################

            raise Exception('utils.live_inference.live_regression() not implemented!') # delete me

            # convert frame to tensor

            # normalize frame

            # transfer frame to GPU

            # transfer model output back to CPU

            # get regression points

            # for each regression class

                # get x and y

                # orient x and y on the frame

                # draw prediction on the frame

            #####################################################################################
            # --------------------------- END YOUR IMPLEMENTATION ----------------------------- #
            #####################################################################################

            show_array(frame) # display the frame in JupyterLab

            IPython.display.clear_output(wait=True) # clear the previous frame

    except KeyboardInterrupt: # if interrupted
        print("Video feed stopped.")
        cam.release() # release the camera feed

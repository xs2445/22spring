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
import numpy as np

from .data_collection import DATA_SHAPE
from .datasets import DEFAULT_TRANSFORMS

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

            # convert frame to tensor
#             frame = torch.tensor(frame)
            frame_d = DEFAULT_TRANSFORMS(frame)

            # normalize the tensor
#             frame = frame / 255.

            # transfer the frame tensor to the GPU
            frame_d = frame_d.to(device)

            # get model output from GPU -> CPU and convert to array
            output = model(frame_d.unsqueeze(0))
#             output = output.detach().numpy()

            # get the predicted class and class_name
            _, pred = torch.max(output.data, 1)
            class_name = class_names[pred]
#             output = output.detatch().numpy()
            

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

            # convert frame to tensor
#             frame = torch.tensor(frame)
            
            # normalize frame
            frame = DEFAULT_TRANSFORMS(frame)
#             frame = frame.resize(1,*frame.size)
            

            # transfer frame to GPU
            frame = frame.to(device)

            # transfer model output back to CPU
            output = model(frame.unsqueeze(0))
            output = output.cpu()
            
            # get regression points
            output = output.detach().numpy().astype(np.int32)
            print(output, output.shape)

            # for each regression class
            for i in regression_class_names:
                # get x and y
                x = output[0,i*2]
                y = output[0,i*2+1]
                # orient x and y on the frame
                x = (x/2+1/2) * DATA_SHAPE[0]
                y = (x/2+1/2) * DATA_SHAPE[1]

                # draw prediction on the frame
                cv2.circle(img, (x, y), 4, COLOR, 2)
            #####################################################################################
            # --------------------------- END YOUR IMPLEMENTATION ----------------------------- #
            #####################################################################################

            show_array(frame) # display the frame in JupyterLab

            IPython.display.clear_output(wait=True) # clear the previous frame

    except KeyboardInterrupt: # if interrupted
        print("Video feed stopped.")
        cam.release() # release the camera feed

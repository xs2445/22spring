"""
This file contains utility functions for performing inference.

E6692 Spring 2022

YOU DO NOT NEED TO MODIFY THIS FUNCTION TO COMPLETE THE ASSIGNMENT
"""
import cv2
import matplotlib.pyplot as plt
import io
from PIL import Image
import IPython
import ast
import torch
import numpy as np
import time

from .torch_utils import detect
from .utils import plot_boxes_cv2


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


def image_inference(image_path, model, conf_thresh, nms_thresh, class_names=None):
    """
    Performs inference on a single image and displays the result in Jupyter cell

    params:
        image_path (string): path to image file
        model (PyTorch model): model with a .detect() method
        conf_thresh (float): 0 - 1 confidence threshold for displaying detections
        nms_thresh (fload): 0 - 1 non-maximum suppression threshold for displaying detections
        class_names (dict): dictionary of class names and class indices
    """

    im = cv2.imread(image_path) # load image
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB) # convert to RGB
    im_resized = cv2.resize(im, (model.width, model.height)) # resize image
    out = detect(model, im_resized, conf_thresh, nms_thresh) # forward pass
    im = plot_boxes_cv2(im, out[0], class_names=class_names) # plot detections
    plt.figure(figsize=(15, 15)) # display detected image
    plt.imshow(im)
    plt.axis('off')
    plt.show()


def webcam_inference(model, class_names=None):
    cam = cv2.VideoCapture(0) # define camera stream

    try:
        print("Video feed started.")

        while True:
            _, frame = cam.read() # read frame from video stream
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # convert raw frame from BGR to RGB

            frame = frame[:, 280:1000, :] # crop and resize
            frame = cv2.resize(frame, (model.width, model.height))

            out = detect(model, frame, 0.2, 0.5)
            frame = plot_boxes_cv2(frame, out[0], class_names=class_names)

            show_array(frame) # display the frame in JupyterLab

            IPython.display.clear_output(wait=True) # clear the previous frame

    except KeyboardInterrupt: # if interrupted
        print("Video feed stopped.")
        cam.release() # release the camera feed


def get_class_names(classes_filename):
    """
    Returns a dictionary of class names and class indices
    where an entry is of the form: { class_index : class_name }

    classes_filename (string): path to the classes text file.
    """

    class_names = {}

    with open(classes_filename, 'r') as names:
        class_names_offset = ast.literal_eval(names.read())
        for key, value in class_names_offset.items():
            class_names.update({ key - 1 : value})

    return class_names


def measure_throughput(model, input_shape=(1, 3, 512, 512), warmup_iterations=50, iterations=1000, verbose=True):
    """
    Measure the throughput of a model with random data.

    params:
        model: PyTorch model
        input_shape (tuple of ints): the input shape of the measurement
        warmup_iterations (int): the number of iterations to "warm up" the GPU before measurement
        iterations (int): the number of measured inferences
        verbose (bool): if true, feedback is printed to the console

    returns:
        throughput (float): the average throughput of the measurements in frames/sec
    """

    input_data = torch.randn(input_shape)
    input_data = input_data.to("cuda")

    if verbose:
        print("Warm up ...")
    with torch.no_grad():
        for _ in range(warmup_iterations):
            features = model(input_data)
    torch.cuda.synchronize()
    if verbose:
        print("Start timing ...")
    timings = []
    with torch.no_grad():
        for i in range(1, iterations + 1):
            start_time = time.time()
            pred_loc = model(input_data)
            torch.cuda.synchronize()
            end_time = time.time()
            timings.append(end_time - start_time)
            if verbose:
                if i % 10 == 0:
                    print('Iteration %d/%d, avg batch time %.2f ms'%(i, iterations, np.mean(timings) * 1000))

    throughput = input_shape[0] / np.mean(timings)

    if verbose:
        print("Input shape:", input_data.size())
        print('Average throughput: %.2f images/second'%(throughput))

    return throughput


def plot_execution_times(throughput_torch, throughput_jit, title,
                         batch_sizes,
                         marker_size=7, logy=True, figsize=(8, 7)):
    """
    Plots the throughput of

    params:
        throughput_torch, throughput_jit (list of floats): throughputs
        title (string): the title of the plot.
        marker_size=7 (int): size of point on plot
        logy=True (bool): if true the y axis is log scaled, else linear
        figsize=(8, 7): the figure dimensions
    """
    fig = plt.figure(figsize=figsize) # make plot
    axis = fig.add_axes([0,0,1,1])
    if logy:
        axis.set_yscale('log')
    axis.plot(throughput_torch, color='red', marker='o', ms=marker_size)
    axis.plot(throughput_jit, color='blue', marker='o', ms=marker_size)

    plt.xticks([int(i) for i in range(len(batch_sizes))], batch_sizes)
    axis.set_ylabel("Throughput (FPS)")
    axis.set_xlabel("Batch Size")

    axis.set_title(title)
    axis.grid()

    axis.legend(["PyTorch", "PyTorch JIT"])
    plt.show()

"""
Utility functions for Lab-TensorRTAndProfiling.

E6692 Spring 2022

YOU DO NOT NEED TO MODIFY THIS FUNCTION TO COMPLETE THE ASSIGNMENT
"""

import torch
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda
import numpy as np
from random import randint
import cv2

from .dataloaders import get_val_loader

def gigs(val):
    """
    Returns the size of val Gigabytes
    
    param:
        val (int): number of gigabytes
        
    returns:
        the number of bytes
    """
    return val * 1 << 30


def load_numpy_weights(torch_weights_path):
    """
    Load the trained PyTorch weight state dictionary with weights as numpy arrays.
    
    param:
        torch_weights_path (string): path to the pytorch weights
        
    returns:
        state_dict (dict): dictionary of weights where an entry is 
                           of the form 
                           
                   { 'weight_name' : weight_array (np.array)}
    """
    state_dict = torch.load(torch_weights_path) # load the weights file
    for param_name, value in state_dict.items(): # convert each weight to numpy array
        state_dict[param_name] = value.cpu().numpy()
        
    return state_dict


# Allocates all buffers required for an engine, i.e. host/device inputs/outputs.
def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream


# Simple helper data class that's a little nicer to use than a 2-tuple.
class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()
    
    
def load_random_test_case(pagelocked_buffer):
    """
    Loads a random validation example to the input buffer
    """
    # Select an image at random to be the test case.
    image_array, label = get_random_testcase()
    # Copy to the pagelocked input buffer
    np.copyto(pagelocked_buffer, image_array)
    return label


def get_random_testcase():
    """
    Returns a random image and corresponding label from the validation set
    """
    val_loader = get_val_loader()
    data, target = next(iter(val_loader))
    random_index = randint(0, len(data) - 1)
    test_image = data.numpy()[random_index].ravel().astype(np.float32)
    test_label = target.numpy()[random_index]
    return test_image, test_label


def do_inference(context, bindings, inputs, outputs, stream):
    """
    inputs and outputs are expected to be lists of HostDeviceMem objects
    """
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]


def save_serialized_engine(trt_engine, filename):
    print("Saving serialized TRT engine.")
    with open(filename, 'wb') as engine_file:
        engine_file.write(trt_engine)
        
        
def load_serialized_engine(filename):
    print("Loading deserialized TRT engine.")
    with open(filename, 'rb') as engine_file:
        return engine_file.read()
    

def draw_bboxes(img, boxes, confs):
    """
    Draw detected bounding boxes on the original image.
    """
    for bb, cf in zip(boxes, confs):
        x_min, y_min, x_max, y_max = bb[0], bb[1], bb[2], bb[3]
        color = (0, 255, 0)
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, 4)
        txt_loc = (max(x_min+2, 0), max(y_min+2, 0))

    return img

def draw_torch_bboxes(img, boxes):
    """
    Draw detected bounding boxes on the original image.
    """
    for bb in boxes:
        x_min, y_min, x_max, y_max = bb[0], bb[1], bb[2], bb[3]
        color = (0, 255, 0)
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, 4)
        txt_loc = (max(x_min+2, 0), max(y_min+2, 0))

    return img

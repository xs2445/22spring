"""
This file contains the function generate_serialized_trt_engine(), which defines the 
TensorRT model network for inference optimization.

E6692 Spring 2022
"""
import torch
import tensorrt as trt
import sys
import numpy as np

from utils.utils import gigs

INPUT_NAME = "data" # name of the input tensor/buffer
INPUT_SHAPE = (1, 1, 28, 28) # the input shape
OUTPUT_NAME = "prob" # name of the output tensor/buffer
OUTPUT_SIZE = 10 # number of output classes
DTYPE = trt.float32 # datatype

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

def generate_serialized_trt_engine(weights):
    """
    This function reimplements the MnistClassifier model structure using
    the Python TensorRT API. Once the TRT network is defined, it is built
    and serialized. During the build phase, TRT performs inference 
    optimizations through reduced precision calculations, layer/tensor fusion,
    kernel auto-tuning, dynamic tensor memory, multi-stream execution, and
    time fusion. These optimizations are performed automatically, given the 
    network is defined correctly. 
    
    To incrementally ensure that your implementation is correct, use "assert" to 
    verify the layer additions to the network are successful. After inserting each
    layer in the network, there should be an assert command.
    
    Visit the links above "builder" and "network" for information on the class methods
    you will need to use.
    
    params:
        weights (dict): the pytorch trained weight dictionary
        
    returns:
        serialized_engine (trt.IHostMemory): pointer class to the serialized trt model/engine
    """
    # initialize the builder, network, and config using explicit batch size
    # https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/infer/Core/Builder.html#builder
    builder = trt.Builder(TRT_LOGGER)
    
    # define the network
    # https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/infer/Graph/Network.html#inetworkdefinition
    network = builder.create_network(EXPLICIT_BATCH)
    
    config = builder.create_builder_config()  
    
    #####################################################################################
    # --------------------------- YOUR IMPLEMENTATION HERE ---------------------------- #
    #####################################################################################

    raise Exception('tensorRTCNN.generate_serialized_trt_engine() not implemented!') # delete me

    #####################################################################################
    # --------------------------- END YOUR IMPLEMENTATION ----------------------------- #
    #####################################################################################
    
    return serialized_engine

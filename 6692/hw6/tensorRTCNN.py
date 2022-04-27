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

# need to first create a logger before create a builder
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
# EXPLICIT_BATCH flag is required in order to import models using the ONNX parser
# refer to https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#explicit-implicit-batch
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
    
    input_tensor = network.add_input(name=INPUT_NAME, dtype=DTYPE, shape=INPUT_SHAPE)
    
    conv1 = network.add_convolution(input=input_tensor, 
                                    num_output_maps=20, kernel_shape=(5,5),
                                    kernel=weights["conv1.weight"],
                                    bias=weights["conv1.bias"])
    conv1.stride = (1,1)
    
    pool1 = network.add_pooling(input=conv1.get_output(0), 
                                type=trt.PoolingType.MAX, 
                                window_size=(2,2))
    pool1.stride = (2,2)
    
    conv2 = network.add_convolution(input=pool1.get_output(0), 
                                    num_output_maps=50, kernel_shape=(5,5),
                                    kernel=weights["conv2.weight"],
                                    bias=weights["conv2.bias"])
    conv2.stride = (1,1)
    
    pool2 = network.add_pooling(input=conv2.get_output(0), 
                                type=trt.PoolingType.MAX, 
                                window_size=(2,2))
    pool2.stride = (2,2)
    
    fc1 = network.add_fully_connected(input=pool2.get_output(0), 
                                      num_outputs=500, 
                                      kernel=weights["fc1.weight"], 
                                      bias=weights["fc1.bias"])
    
    relu1 = network.add_activation(input=fc1.get_output(0), type=trt.ActivationType.RELU)
    
    fc2 = network.add_fully_connected(input=relu1.get_output(0), 
                                      num_outputs=OUTPUT_SIZE, 
                                      kernel=weights["fc2.weight"], 
                                      bias=weights["fc2.bias"])
    
    fc2.get_output(0).name = OUTPUT_NAME
    
    network.mark_output(fc2.get_output(0))
    
#     serialized_engine = builder.build_serialized_network(network=network, config=config)
    engine = builder.build_engine(network=network, config=config)
    serialized_engine = engine.serialize()
    
    #####################################################################################
    # --------------------------- END YOUR IMPLEMENTATION ----------------------------- #
    #####################################################################################
    
    return serialized_engine

"""
This script generates a serialized TensorRT engine for the MNIST Classifier.

E6692 Spring 2022

YOU DO NOT NEED TO MODIFY THIS FILE TO COMPLETE THE ASSIGNMENT
"""
import numpy as np

from utils.utils import load_numpy_weights, save_serialized_engine, allocate_buffers, load_random_test_case, do_inference
from tensorRTCNN import generate_serialized_trt_engine

TRT_MNIST_ENGINE = './engines/trt_model.engine'

def serializeMNIST(torch_mnist_weights, trt_mnist_engine):
    """
    Load the Pytorch weight dictionary as numpy arrays and call
    generate_serialized_trt_engine() to create the serialized 
    TensorRT engine. The serialized engine is saved to a file
    with save_serialized_engine().
    
    params:
        torch_mnist_weights (string): path to pytorch weights
        trt_mnist_engine (string): path to save serialized trt engine
    """
    # load PyTorch MNIST classifier weights as numpy arrays
    print("Loading PyTorch weights.")
    pytorch_weights = load_numpy_weights(torch_mnist_weights)
    
    # generate trt engine 
    serialized_trt_engine = generate_serialized_trt_engine(pytorch_weights)
    
    # save the serialized trt engine
    save_serialized_engine(serialized_trt_engine, trt_mnist_engine)

    
def trt_prediction(deserialized_engine):
    """
    Make a prediction with the TRT engine. load_random_test_case() 
    chooses a random image from the validation set and loads it into
    the input buffer.
    
    param:
        deserialized_engine (trt.ICudaEngine) : the deserialized TRT engine
        
    returns:
        ground_truth (int): the ground truth label (0-9)
        prediction (int): the prediction of the TRT model (0-9)
    """
    # allocate memory on the GPU for input and output
    inputs, outputs, bindings, stream = allocate_buffers(deserialized_engine)
    
    # generate the execution context
    context = deserialized_engine.create_execution_context()
    
    # load an image into the input memory buffer and return the label
    ground_truth = load_random_test_case(pagelocked_buffer=inputs[0].host)
    
    # perform inference with trt
    [output] = do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
    
    # get the prediction 
    prediction = np.argmax(output)
    
    return ground_truth, prediction
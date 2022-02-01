"""
This file contains functions for creating 2D convolution visualizations.

E6692 Spring 2022
"""

import numpy as np

def dilate_kernel(kernel, dilation_factor):
    """
    Perform kernel dilation. See here for more details on kernel dilation: https://www.geeksforgeeks.org/dilated-convolution/
    
    params:
        kernel (np.array): the convolutional kernel to be dilated. Shape (M, N)
        dilation_factor (int): the dilation factor. This parameter indicates the spacing
                               of elements in the dilated kernel. The number of zeros inserted
                               between original kernel elements is dilation_factor - 1 in both
                               dimensions.
                               
    returns:
        dilated_kernel (np.array): the dilated kernel. Shape ((M - 1) * dilation_factor + 1, (N - 1) * dilation_factor + 1)
    """
    #####################################################################################
    # --------------------------- YOUR IMPLEMENTATION HERE ---------------------------- #
    #####################################################################################
    
    raise Exception('utils.convolution2D.dilate_kernel() not implemented!') # delete me
    
    #####################################################################################
    # --------------------------- END YOUR IMPLEMENTATION ----------------------------- #
    #####################################################################################
    
    return dilated_kernel


def pad_img(image, dilated_kernel):
    """
    Pad the image with zeros such that convolution is performed on every pixel with every kernel element.
    
    params:
        image (np.array): the image array of shape (rows, cols, channels)
        dilated_kernel (np.array): the dilated kernel of shape (O, P)
        
    returns:
        padded_image (np.array): the padded image
        
    HINT: Use np.pad to simplify this function. https://numpy.org/doc/stable/reference/generated/numpy.pad.html
    """
    #####################################################################################
    # --------------------------- YOUR IMPLEMENTATION HERE ---------------------------- #
    #####################################################################################
    
    raise Exception('utils.convolution2D.pad_img() not implemented!') # delete me
    
    #####################################################################################
    # --------------------------- END YOUR IMPLEMENTATION ----------------------------- #
    #####################################################################################
    
    return padded_image


def calc_conv2d(image, kernel):
    """
    Calculate the 2D convolution between the image and the kernel with a stride of 1. This 
    function should be able to handle both 2D images (1 channel) and 3D images (3 channels).
    You may use numpy functions to implement 2D convolution, but your solution should be
    explicit. i.e. you cannot simply call scipy.signal.convolve2d.
    
    params:
        image (np.array): the input image of shape (A, B) OR (A, B, channels)
        kernel (np.array): the kernel of shape (C, D)
        
    returns:
        outut (np.array): the result of the 2D convolution. Shape: (A - C + 1, B - D + 1)
    """
    
    #####################################################################################
    # --------------------------- YOUR IMPLEMENTATION HERE ---------------------------- #
    #####################################################################################
    
    raise Exception('utils.convolution2D.calc_conv2d() not implemented!') # delete me
    
    #####################################################################################
    # --------------------------- END YOUR IMPLEMENTATION ----------------------------- #
    #####################################################################################

    return output
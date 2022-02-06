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
    
    m ,n = kernel.shape
    # generate a zero dilated kernel 
    dilated_kernel = np.zeros(((m-1)*dilation_factor+1, (n-1)*dilation_factor+1))
    # fill in the original kernel into the dilated kernel
    for i in range(m):
        for j in range(n):
            dilated_kernel[i*dilation_factor,j*dilation_factor] = kernel[i,j]
    
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
    
    m, n = dilated_kernel.shape
    pad_m = m//2
    pad_n = n//2
    padded_image = np.pad(image, ((pad_m,pad_m),(pad_n,pad_n),(0,0)), mode='constant')
    
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
    
    shape_img = image.shape
    shape_knl = kernel.shape

    if len(shape_img) == 3:
        channel = shape_img[2]
    elif len(shape_img) == 2:
        channel = 0
    else:
        raise("Dimensino of input image should be 2 or 3!")
    
    output = np.zeros((shape_img[0]-shape_knl[0]+1, shape_img[1]-shape_knl[1]+1))
    # reverse the kernel to make the code of convolution simple
    kernel_reversed = np.zeros_like(kernel)
    for i in range(shape_knl[0]):
        for j in range(shape_knl[1]):
            kernel_reversed[i,j] = kernel[shape_knl[0]-i-1, shape_knl[1]-j-1]
    
    # The process of correlation. It's convolution because the kernel is reversed.
    for m in range(shape_img[0]-shape_knl[0]+1):
        for n in range(shape_img[1]-shape_knl[1]+1):
            if channel:
                for c in range(channel):
                    output[m, n] += np.sum(image[m:m+shape_knl[0],n:n+shape_knl[1],c] * kernel_reversed)
            else:
                output[m, n] = np.sum(image[m:m+shape_knl[0],n:n+shape_knl[1]] * kernel_reversed)


    #####################################################################################
    # --------------------------- END YOUR IMPLEMENTATION ----------------------------- #
    #####################################################################################

    return output
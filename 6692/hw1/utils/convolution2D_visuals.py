"""
This file contains visualization utility functions for displaying 2D convolution and
dataset generation functions.

E6692 Spring 2022

YOU DO NOT NEED TO MODIFY THESE FUNCTIONS TO COMPLETE THE ASSIGNMENT
"""

import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from .convolution2D import *


def get_cifar10(data_path='./data'):
    """
    Download and verify CIFAR 10 Image dataset. 
    """
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
                                  )

    return torchvision.datasets.CIFAR10(root=data_path, download=True, transform=transform)
    
    
def display_image(image):
    """
    Display the image in the Jupyter Notebook.
    """
    image = image / 2 + 0.5 # unnormalize the image
    image = np.transpose(image.numpy(), (1, 2, 0))
    plt.figure(figsize=(8, 8))
    plt.imshow(image)
    plt.axis('off')
    plt.show()
    
    
def plot_ax(fig, ax, n_row, n_col, image, title, ylabel = '', image_type='in'):
    """
    plots the image of a subplot
        
    Args:
        fig (figure): Figure object of the total plot
        ax (axis): Axis object
        n_row (integer): Row index, on which subaxis the image should be plotted
        n_col (integer): Col index, on which subaxis the image should be plotted
        image (np array): Image to plot
        title (str): Title of the subplot
        ylabel: Label of the Y-Axis
        image_type (str): one of {'in', 'kernel', or 'out'} used to configure axes.
    """

    ax[n_row, n_col].xaxis.set_ticklabels([])
    ax[n_row, n_col].yaxis.set_ticklabels([])
    ax[n_row, n_col].set_title(title)
    ax[n_row, n_col].set_ylabel(ylabel)
    if image_type == 'out':
#         image = (image - np.min(image)) / (np.max(image) - np.min(image))
        image = np.clip(image, 0, 1)
        
    if image_type == 'kernel':
        cmap = 'copper'
    else:
        cmap = None
        
    im1 = ax[n_row, n_col].imshow(image, cmap=cmap)
    divider = make_axes_locatable(ax[n_row, n_col])
    
    if image_type == 'kernel':
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im1, cax=cax, orientation='vertical')
    
    
    
def visualize(image, kernel, dilutions = [0]):
    """
    visualizes different diluted convolutions for an image and kernel
    
    Args:
        image (np_array): Image to plot
        kernel (np_array): Kernel values before dilution
        dilutions (list): List of how many steps applied
    """
    image = image / 2 + 0.5 # unnormalize image
    plt.rcParams['figure.figsize'] = [12, 12]
    fig, ax = plt.subplots(nrows = max(len(dilutions)+1,2), ncols = 3)
    plot_ax(fig, ax, 0, 0, image, 'Raw image', image_type='in')
    plot_ax(fig, ax, 0, 1, kernel, 'Raw kernel', image_type='kernel')
    plot_ax(fig, ax, 0, 2, image, 'Raw image', image_type='in')
    ax[0, 2].axis('off')
    i = 1
    for dilution in dilutions:
        diluted_kernel = dilate_kernel(kernel, dilution)
        padded_img = pad_img(image, diluted_kernel)
        conved_img = calc_conv2d(padded_img, diluted_kernel)
        plot_ax(fig, ax, i, 0, padded_img, 'Padded image', 'Dilution: ' + str(dilution), image_type='in')
        plot_ax(fig, ax, i, 1, diluted_kernel, 'Diluted kernel', image_type='kernel')
        plot_ax(fig, ax, i, 2, conved_img, 'Output', image_type='out')
        i += 1
    
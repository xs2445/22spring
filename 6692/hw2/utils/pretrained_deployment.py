"""
This file contains helper functions for the pretrained deployment portion of Lab-JetsonNanoSetup-PretrainedDeployment.

E6692 Spring 2022

YOU DO NOT NEED TO MODIFY THESE FUNCTIONS TO COMPLETE THE ASSIGNMENT
"""
import torch
import os
import io
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import IPython
from IPython.display import display

from .install_dependancies import install_dependancies

try:
    from google_images_download import google_images_download
except:
    install_dependancies()
    from google_images_download import google_images_download
    

def download_n_images(query, n):
    """
    Download the first N google images into the "downloads" folder
    of the root directory. Images are organized into folders titled by query.
    
    params:
        query (string) : google image query
        n (int) : number of images to retrieve
    """
    response = google_images_download.googleimagesdownload() 
    
    arguments = {"keywords" : query,
                 "format" : "jpg",
                 "limit" : n,
                 "print_urls" : True,
                 "size" : "medium",
                 "aspect_ratio" : "panoramic"}
    
    try:
        response.download(arguments)
      
    # Handling File NotFound Error    
    except FileNotFoundError: 
        arguments = {"keywords" : query,
                     "format" : "jpg",
                     "limit" : n,
                     "print_urls" : True, 
                     "size" : "medium"}
                       
        # Providing arguments for the searched query
        try:
            # Downloading the photos based
            # on the given arguments
            response.download(arguments) 
        except:
            pass
        
        
def display_image(image):
    """
    Display an image in Jupyter Notebook.
    
    param:
        image: numpy array representing an image, PIL image, or a path to an image
    """
    if isinstance(image, str):
        image_array = cv2.imread(image)
        image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
        image_array = Image.fromarray(image_array)
        
    elif isinstance(image, np.ndarray):
        image_array = image
        
    else:
        raise Exception('image type is not supported.')
    
    plt.figure(figsize=(12, 8))
    plt.imshow(image_array)
    plt.axis('off')
    plt.show()
    
    
def show_array(a, fmt='jpeg'):
    """
    Display array using ipython widget.
    """
    f = io.BytesIO()
    Image.fromarray(a).save(f, fmt)
    display(IPython.display.Image(data=f.getvalue()))
    
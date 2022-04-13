"""
Google Images Download.

E6692 Spring 2022

YOU DO NOT NEED TO MODIFY THIS FUNCTION TO COMPLETE THE ASSIGNMENT
"""
import os

DEPENDANCIES = ['tqdm',
                'git+https://github.com/Joeclinton1/google-images-download.git']

def install_dependancies():
    """
    Install the dependancies listed above with pip.
    """
    for dependancy in DEPENDANCIES:
        os.system('pip3 install {}'.format(dependancy))
        print("{} installed.".format(dependancy))
        
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
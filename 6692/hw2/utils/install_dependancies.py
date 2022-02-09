"""
Install dependancies function for Lab-JetsonNanoSetup-PretrainedDeployment.

E6692 Spring 2022

YOU DO NOT NEED TO MODIFY THIS FUNCTION TO COMPLETE THE ASSIGNMENT
"""
import os

DEPENDANCIES = ['facenet_pytorch', 
                'tqdm',
                'git+https://github.com/Joeclinton1/google-images-download.git']

def install_dependancies():
    """
    Install the dependancies listed above with pip.
    """
    for dependancy in DEPENDANCIES:
        os.system('pip3 install {}'.format(dependancy))
        print("{} installed.".format(dependancy))

    
"""
This file contains classes for defining custom CNN classification and regression
models.

E6692 Spring 2022
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomClassifier(nn.Module):
    """
    A custom CNN classification model class. This class inherits from nn.Module,
    which defines the __call__ method and allows us to train custom models easily.
    We redefine the forward() function to design a custom model. It is recommended
    that you use the nn.functional layers for your implementation.

    Your basic CNN classification model should look something like this:

    input -> convolutional layers -> fully connected layers -> output

    where a convolutional layer is comprised of:

    2d convolution -> activation -> pooling

    and a fully connected layer is comprised of:

    linear (affine) layer -> activation
    """
    def __init__(self, num_classes):
        """
        Initialize the parent class nn.Module and the CustomClassifier. Layers
        of the model should be defined here.

        param:
            num_classes (int): the number of classes to predict
        """

        super().__init__()

        #####################################################################################
        # --------------------------- YOUR IMPLEMENTATION HERE ---------------------------- #
        #####################################################################################

        raise Exception('utils.models.CustomClassifier.__init__() not implemented!') # delete me

        # define layers here

        #####################################################################################
        # --------------------------- END YOUR IMPLEMENTATION ----------------------------- #
        #####################################################################################


    def forward(self, x):
        """
        The forward pass of the model. This is where the layers defined in __init__
        are called in sequence.

        param:
            x (torch.Tensor): an input tensor.

        returns:
            x (torch.Tensor): the output tensor
        """

        #####################################################################################
        # --------------------------- YOUR IMPLEMENTATION HERE ---------------------------- #
        #####################################################################################

        raise Exception('utils.models.CustomClassifier.forward() not implemented!') # delete me

        # define forward pass of model using PyTorch functional API

        #####################################################################################
        # --------------------------- END YOUR IMPLEMENTATION ----------------------------- #
        #####################################################################################

        return x


class CustomRegression(nn.Module):
    """
    A custom CNN regression model class. Very similar to CustomClassifier, see
    above for documentation.
    """

    def __init__(self, num_classes):
        """
        Initialize the parent class nn.Module and the CustomRegression. Layers
        of the model should be defined here.

        param:
            num_classes (int): the number of regression classes
        """
        super().__init__()

        #####################################################################################
        # --------------------------- YOUR IMPLEMENTATION HERE ---------------------------- #
        #####################################################################################

        raise Exception('utils.models.CustomRegression.__init__() not implemented!') # delete me

        # define layers here

        #####################################################################################
        # --------------------------- END YOUR IMPLEMENTATION ----------------------------- #
        #####################################################################################


    def forward(self, x):
        """
        The forward pass of the model. This is where the layers defined in __init__
        are called in sequence.

        param:
            x (torch.Tensor): an input tensor.

        returns:
            x (torch.Tensor): the output tensor
        """
        #####################################################################################
        # --------------------------- YOUR IMPLEMENTATION HERE ---------------------------- #
        #####################################################################################

        raise Exception('utils.models.CustomRegression.forward() not implemented!') # delete me

        # define forward pass of model using PyTorch functional API

        #####################################################################################
        # --------------------------- END YOUR IMPLEMENTATION ----------------------------- #
        #####################################################################################

        return x

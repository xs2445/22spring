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
        
        self.model = nn.Sequential(
<<<<<<< HEAD
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1),
            nn.ReLU(),
#             nn.Dropout(p=0.4),
            nn.AvgPool2d(kernel_size=2),
            
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1),
            nn.ReLU(),
#             nn.Dropout(p=0.4),
            nn.AvgPool2d(kernel_size=2),
            
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
#             nn.Dropout(p=0.4),
            nn.AvgPool2d(kernel_size=2),
            
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
#             nn.Dropout(p=0.4),
            nn.AvgPool2d(kernel_size=2),
            
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1),
            nn.ReLU(),
#             nn.Dropout(p=0.4),
            nn.AvgPool2d(kernel_size=2),
            
            nn.Flatten(),
            
            nn.Linear(in_features=4608, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=num_classes),
            nn.ReLU()
=======
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.Flatten(),
            
            nn.Linear(in_features=120, out_features=84),
            nn.Tanh(),
            nn.Linear(in_features=84, out_features=num_classes),
>>>>>>> 6fe9c41fbf30ffe8cbd0de3d80c4fb35aacd7459
#             F.softmax()
            
        )


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



        # define forward pass of model using PyTorch functional API
        
        x = self.model(x)
<<<<<<< HEAD
#         x = F.softmax(x)
=======
        x = F.softmax(x)
>>>>>>> 6fe9c41fbf30ffe8cbd0de3d80c4fb35aacd7459

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

        # define layers here
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1),
            nn.ReLU(),
#             nn.Dropout(p=0.4),
            nn.AvgPool2d(kernel_size=2),
            
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1),
            nn.ReLU(),
#             nn.Dropout(p=0.4),
            nn.AvgPool2d(kernel_size=2),
            
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
#             nn.Dropout(p=0.4),
            nn.AvgPool2d(kernel_size=2),
            
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
#             nn.Dropout(p=0.4),
            nn.AvgPool2d(kernel_size=2),
            
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1),
            nn.ReLU(),
#             nn.Dropout(p=0.4),
            nn.AvgPool2d(kernel_size=2),
            
            nn.Flatten(),
            
            nn.Linear(in_features=4608, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=num_classes*2),
            nn.ReLU()
#             F.softmax()
            
        )

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

        # define forward pass of model using PyTorch functional API
        x = self.model(x)

        #####################################################################################
        # --------------------------- END YOUR IMPLEMENTATION ----------------------------- #
        #####################################################################################

        return x

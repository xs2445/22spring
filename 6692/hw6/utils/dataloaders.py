"""
This file functions for loading/downloading the MNIST training and validation sets.

E6692 Spring 2022

YOU DO NOT NEED TO MODIFY THIS FUNCTION TO COMPLETE THE ASSIGNMENT
"""

import torch
from torchvision import datasets, transforms

TRANSFORM = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
BATCH_SIZE = 64

def get_train_loader():
    # load the training set DataLoader
    return torch.utils.data.DataLoader(
                datasets.MNIST('/tmp/mnist/data', train=True, download=True, transform=TRANSFORM),
                batch_size=BATCH_SIZE,
                shuffle=True,
                num_workers=1,
                timeout=600)

def get_val_loader():
    # load the validation set DataLoader
    return torch.utils.data.DataLoader(
                datasets.MNIST('/tmp/mnist/data', train=False, transform=TRANSFORM),
                batch_size=BATCH_SIZE,
                shuffle=True,
                num_workers=1,
                timeout=600)
    
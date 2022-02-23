"""
This file contains PyTorch training functions for classification and regression
tasks.

E6692 Spring 2022
"""
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

def train_classification(model, device, train_dataset, val_dataset, batch_size, optimizer, epochs, save_path):
    """
    This function defines the training process for a classification model.
    Refer to this for help and how to use PyTorch optimization functions properly:
    https://pytorch.org/tutorials/beginner/introyt/trainingyt.html

    params:
        model (nn.Module): the PyTorch classification model

        device (torch.device): GPU specification

        train_dataset (torch.utils.data.Dataset): the classification training dataset

        val_dataset (torch.utils.data.Dataset): the classification validation dataset

        batch_size (int): the batch size. This determines how many iterations it
                          takes to go through an epoch and how many images are included
                          in the gradient calculation at one time.

        optimizer (torch.optim): the optimization function (i.e. Adam, SGD, RMSprop, etc.)

        epochs (int): the number of passes through the entire dataset

        save_path (string): path to save the model with the highest validation
                            accuracy.

    returns:
        train_loss_history (list of floats): the training loss by iteration

        train_accuracy_history (list of floats): the training accuracy by iteration

        val_loss_history (list of floats): the validation loss by iteration

        val_accuracy_history (list of floats): the validation accuracy by iteration

    """
    #####################################################################################
    # --------------------------- YOUR IMPLEMENTATION HERE ---------------------------- #
    #####################################################################################

    # clear GPU cache
    torch.cuda.empty_cache() 

    # define training and validation data loaders
    

    # initialize empty lists for storing training and validation loss and accuracy
    train_loss_history = []
    val_loss_history = []
    # initialize highest validation accuracy value
    
    # iterate through epochs

        # initialize training loss and iteration values

        # make sure model is in train mode (dropout activated, weights not frozen, etc.)

        # iterate through training epoch

            # transfer images and labels to GPU

            # set the gradients of batch to zero

            # complete forward pass of batch through model

            # calculate the loss of the output

            # compute the gradients of the pass

            # update the model weights

            # update the training loss and accuracy histories

        # initialize validation loss and iteration values

        # switch model to evaluation mode

        # iterate through validation epoch

            # transfer images and labels to GPU

            # complete forward pass of batch through model

            # calculate loss

            # update the validation loss and accuracy histories

        # if validation accuracy is higher than previous highest, save model weights and update highest val accuracy value

        # print losses and accuracies each epoch

    #####################################################################################
    # --------------------------- END YOUR IMPLEMENTATION ----------------------------- #
    #####################################################################################

    # return training and validation loss and accuracy histories
    return train_loss_history, train_accuracy_history, val_loss_history, val_accuracy_history


def train_regression(model, device, train_dataset, val_dataset, batch_size, optimizer, epochs, save_path):
    """
    This function defines the training process for a regression model.
    Refer to this for help and how to use PyTorch optimization functions properly:
    https://pytorch.org/tutorials/beginner/introyt/trainingyt.html

    Note on Loss function: you will need to implement MSE loss manually since we
    store one regression point label to one image.

    params:
        model (nn.Module): the PyTorch regression model

        device (torch.device): GPU specification

        train_dataset (torch.utils.data.Dataset): the regression training dataset

        val_dataset (torch.utils.data.Dataset): the regression validation dataset

        batch_size (int): the batch size. This determines how many iterations it
                          takes to go through an epoch and how many images are included
                          in the gradient calculation at one time.

        optimizer (torch.optim): the optimization function (i.e. Adam, SGD, RMSprop, etc.)

        epochs (int): the number of passes through the entire dataset

        save_path (string): path to save the model with the highest validation
                            accuracy.

    returns:
        train_loss_history (list of floats): the training loss by iteration

        val_loss_history (list of floats): the validation loss by iteration
    """

    #####################################################################################
    # --------------------------- YOUR IMPLEMENTATION HERE ---------------------------- #
    #####################################################################################

#     raise Exception('utils.training.train_classification() not implemented!') # delete me

    # clear GPU cache

    # define training and validation data loaders

    # initialize empty list to store loss histories

    # initialize lowest validation loss value with some high value

    # iterate through epochs

        # initialize training loss and iteration values

        # make sure model is in train mode (dropout activated, weights not frozen, etc.)

        # iterate through training epoch

            # transfer images and labels to GPU

            # set the gradients of batch to zero

            # complete forward pass of batch through model

            # initialize batch training loss value

            # iterate through batch and calculate MSE loss of relevant predicted regression point
            # since our data is stored as one image to one regression point, make sure you are
            # calculating the loss only for the prediction that corresponds to the label.

            # average training loss accross batch size

            # compute the gradients of the pass

            # update the model weights

            # update the training loss history

        # initialize validation loss and iteration values

        # switch model to evaluation mode

        # iterate through validation epoch

            # transfer images and labels to GPU

            # initialize batch validation loss value

            # iterate through batch and calculate MSE loss of relevant predicted regression point
            # since our data is stored as one image to one regression point, make sure you are
            # calculating the loss only for the prediction that corresponds to the label.

            # average validation loss accross batch size

            # update the validation loss history

        # if validation loss is lower than previous lowest, save model weights and update lowest val loss value

        # print losses for each epoch

    #####################################################################################
    # --------------------------- END YOUR IMPLEMENTATION ----------------------------- #
    #####################################################################################

    # return training and validation loss histories
    return train_loss_history, val_loss_history


def plot_loss_and_accuracy(loss, accuracy, title, labels):
    """
    This function plots the loss and accuracy histories on the same plot and
    displays it in the jupyter cell output.

    params:
        loss (list of float): loss history
        accuracy (list of float): accuracy history
        title (string): title of the plot
        labels (list of string): list of plot labels
    """
    _, ax1 = plt.subplots(figsize=(10, 8))
    ax2 = ax1.twinx()

    ax1.plot(loss, color='blue')
    ax2.plot(accuracy, color='orange')
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Loss")
    ax1.legend([labels[1]], loc='best')
    ax2.set_ylabel("Accuracy")
    ax2.legend([labels[0]], loc='best')

    plt.title(title)
    plt.show()

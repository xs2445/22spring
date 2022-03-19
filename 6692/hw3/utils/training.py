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
<<<<<<< HEAD
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size)
    
    # number of samples and batches
    n_sample_train = train_loader.dataset.__len__()
    n_batch_train = len(train_loader)
    n_sample_val = val_loader.dataset.__len__()
    n_batch_val = len(val_loader)
=======
    
>>>>>>> 6fe9c41fbf30ffe8cbd0de3d80c4fb35aacd7459

    # initialize empty lists for storing training and validation loss and accuracy
    train_loss_history = []
    val_loss_history = []
<<<<<<< HEAD
    train_accuracy_history = []
    val_accuracy_history = []
    
    # initialize highest validation accuracy value
    acc_val_high = 0
    
    # loss function
    loss_fn = torch.nn.CrossEntropyLoss()
=======
    # initialize highest validation accuracy value
>>>>>>> 6fe9c41fbf30ffe8cbd0de3d80c4fb35aacd7459
    
    # iterate through epochs
    for epoch in range(epochs):
        
        print("Epoch:", epoch+1)
    
        # initialize training loss and iteration values
        train_running_loss = 0
        train_running_acc = 0
        count = 0
        
        # make sure model is in train mode (dropout activated, weights not frozen, etc.)
        model.train(True)
        
        # iterate through training epoch
        # passing a batch of data to the model in each iteration
        for images, labels in iter(train_loader):

            # transfer images and labels to GPU
            images = images.to(device)
            labels = labels.to(device)
            # set the gradients of batch to zero
            optimizer.zero_grad()
            
            # complete forward pass of batch through model
            logits = model(images)
            
            # calculate the loss of the output
            loss = loss_fn(logits, labels)
            
            # compute the gradients of the pass
            loss.backward()

            # update the model weights
            optimizer.step()

            # update the training loss and accuracy histories
            # sum of loss for all sample
            count += 1
            train_running_loss += loss.item()
            # predictions
            _, pred = torch.max(logits.data, 1)
            # number of correct predictions in this batch
            train_running_acc += (labels == pred).sum().item()
            if count % 4 == 3:
                print("sample:{:d}, train_loss:{:.4f}, train_acc:{:.4f}".format(count*batch_size, train_running_loss/(count*batch_size), train_running_acc/(count*batch_size)))
            
            
        # average loss for each sample in that epoch
        train_loss_history.append(train_running_loss/n_sample_train)
        # average accuracy of that epoch
        train_accuracy_history.append(train_running_acc/n_sample_train)
        

        # initialize validation loss and iteration values
        val_running_loss = 0
        val_running_acc = 0
        
        # switch model to evaluation mode
        model.eval()
        
        # iterate through validation epoch
        for images, labels in iter(val_loader):

            # transfer images and labels to GPU
            images = images.to(device)
            labels = labels.to(device)

            # complete forward pass of batch through model
            logits = model(images)
            
            # calculate loss
            loss = loss_fn(logits, labels)

            # update the validation loss and accuracy histories
            val_running_loss += loss.item()
            _, pred = torch.max(logits.data, 1)
            val_running_acc += (labels == pred).sum().item()
            
        val_loss_history.append(val_running_loss/n_sample_val)
        val_accuracy_history.append(val_running_acc/n_sample_val)
            
            
        # print losses and accuracies each epoch
        print("Train Loss:{:.4f}, Acc:{:.4f}".format(train_loss_history[-1], train_accuracy_history[-1]))
        print("Val Loss:{:.4f}, Acc:{:.4f}".format(val_loss_history[-1], val_accuracy_history[-1]))
            
        # if validation accuracy is higher than previous highest, save model weights and update highest val accuracy value
        if val_accuracy_history[-1] > acc_val_high:
            acc_val_high = val_accuracy_history[-1]
            torch.save(model.state_dict(), save_path)
            print("Best model saved.")
        
        print("\n")
        
        


        

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

    # clear GPU cache
    torch.cuda.empty_cache() 
    
    # define training and validation data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size)
    
    # initialize empty list to store loss histories
    train_loss_history = []
    val_loss_history = []
    
    # initialize lowest validation loss value with some high value
    val_loss_low = 100
    
    # number of samples and batches
    n_sample_train = train_loader.dataset.__len__()
    n_batch_train = len(train_loader)
    n_sample_val = val_loader.dataset.__len__()
    n_batch_val = len(val_loader)
    
    # iterate through epochs
    for epoch in range(epochs):
        print("Epoch:", epoch+1)
        
        # initialize training loss and iteration values
        train_running_loss = 0
        train_running_acc = 0
        count = 0
        
        # make sure model is in train mode (dropout activated, weights not frozen, etc.)
        model.train(True)
        
        # iterate through training epoch
        for images, labels, coordinates in iter(train_loader):

            # transfer images and labels to GPU
            images = images.to(device)
            labels = labels.to(device)
            coordinates = coordinates.to(device)

            # set the gradients of batch to zero
            optimizer.zero_grad()

            # complete forward pass of batch through model
            logits = model(images)

            # initialize batch training loss value
            loss = torch.zeros(1)
            loss = loss.to(device)

            # iterate through batch and calculate MSE loss of relevant predicted regression point
            # since our data is stored as one image to one regression point, make sure you are
            # calculating the loss only for the prediction that corresponds to the label.
            for i in range(logits.size(dim=0)):
                logit = logits[i, 2*labels[i] : 2*labels[i]+1]
                loss += torch.mean(torch.square((logit-coordinates[i])))

            # average training loss accross batch size
            loss = loss/logits.size(dim=0)
            
            # compute the gradients of the pass
            loss.backward()

            # update the model weights
            optimizer.step()

            # update the training loss history
            train_running_loss += loss.item()
            count += 1
            
            if count % 10 == 9:
                print("sample:{:d}, train_loss:{:.4f}".format(count*batch_size, train_running_loss/count))
        
        train_loss_history.append(train_running_loss/n_batch_train)

        # initialize validation loss and iteration values
        val_running_loss = 0
            
        # switch model to evaluation mode
        model.eval()

        # iterate through validation epoch
        for images, labels, coordinates in iter(val_loader):

            # transfer images and labels to GPU
            images = images.to(device)
            labels = labels.to(device)
            coordinates = coordinates.to(device)
            
            # initialize batch validation loss value
            loss = torch.zeros(1)
            loss = loss.to(device)
            
            # iterate through batch and calculate MSE loss of relevant predicted regression point
            # since our data is stored as one image to one regression point, make sure you are
            # calculating the loss only for the prediction that corresponds to the label.
            for i in range(logits.size(dim=0)):
                logit = logits[i, 2*labels[i] : 2*labels[i]+1]
                loss += torch.mean(torch.square((logit-coordinates[i])))
                
            # average validation loss accross batch size

            # update the validation loss history
            val_running_loss += loss.item()
        
        val_loss_history.append(val_running_loss/n_sample_val)

        # print losses for each epoch
        print("Train Loss:{:.4f}".format(train_loss_history[-1]))
        print("Val Loss:{:.4f}".format(val_loss_history[-1]))
        
        # if validation loss is lower than previous lowest, save model weights and update lowest val loss value
        if val_loss_history[-1] < val_loss_low:
            val_loss_low = val_loss_history[-1]
            torch.save(model.state_dict(), save_path)
            print("Best model saved.")
            
        print('\n')
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
    ax1.legend([labels[0]], loc='best')
    ax2.set_ylabel("Accuracy")
    ax2.legend([labels[1]], loc='best')

    plt.title(title)
    plt.show()

"""
This file contains classes and functions for training a PyTorch MNIST Model

E6692 Spring 2022

YOU DO NOT NEED TO MODIFY THIS FUNCTION TO COMPLETE THE ASSIGNMENT
"""
# import modules
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import numpy as np
import os
from random import randint

device = 'cuda'

# Define network architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, kernel_size=5)
        self.conv2 = nn.Conv2d(20, 50, kernel_size=5)
        self.fc1 = nn.Linear(800, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.max_pool2d(self.conv1(x), kernel_size=2, stride=2)
        x = F.max_pool2d(self.conv2(x), kernel_size=2, stride=2)
        x = x.view(-1, 800)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# define classifier obejct - contains training function/weight extraction
class MnistClassifier(object):
    def __init__(self, batch_size, learning_rate, sgd_momentum, log_interval):
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.sgd_momentum = sgd_momentum
        self.log_interval = log_interval # how often to calculate validation accuracy
        self.network = Net().to(device)

    # Train the network for one or more epochs, validating when iteration % log_interval == 0
    def learn(self, train_loader, val_loader, num_epochs=1):
        
        train_history = []
        val_history = []
        
        # Train the network for a single epoch - epoch is the epoch index/count
        def train(epoch):
            self.network.train()
            optimizer = optim.SGD(self.network.parameters(), lr=self.learning_rate, momentum=self.sgd_momentum)
            for batch, (data, target) in enumerate(train_loader):
                data, target = Variable(data), Variable(target)
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = self.network(data)
                loss = F.nll_loss(output, target)
                loss.backward()
                train_history.append(loss.data.item())
                optimizer.step()
                if batch % self.log_interval == 0:
                    val_acc = val()
                    val_history.append(val_acc)
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch * len(data), len(train_loader.dataset), 100. * batch / len(train_loader), loss.data.item()))

        # evaluate on the validation set
        def val():
            self.network.eval()
            val_loss = 0
            correct = 0
            for data, target in val_loader:
                with torch.no_grad():
                    data, target = Variable(data), Variable(target)
                    data, target = data.to(device), target.to(device)
                output = self.network(data)
                val_loss += F.nll_loss(output, target).data.item()
                pred = output.data.max(1)[1]
                correct += pred.eq(target.data).cpu().sum()
            val_loss /= len(val_loader)
            val_acc =  100 * correct / len(val_loader.dataset)
            print('\nValidation: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(val_loss, correct, len(val_loader.dataset), val_acc))
            return val_acc
        # train for num_epochs
        for e in range(num_epochs):
            train(e + 1)
            
        # return the training loss and validation accuracy
        return train_history, val_history


    def get_weights(self):
        """
        Returns the model's weight dictionary
        """
        return self.network.state_dict()
    


    
    
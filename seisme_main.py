# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 16:14:47 2021

@author: JuHik
"""

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import MNIST, FashionMNIST

from helper import normalization



#extraction des données

"""
train 111 mesures (105 S_a(T), S_a,avg, delta_t1, delta_t2, delta_v, collapse_cap)

val 111 mesures (105 S_a(T), S_a,avg, delta_t1, delta_t2, delta_v, collapse_cap)

test 110 mesures (105 S_a(T), S_a,avg, delta_t1, delta_t2, delta_v)

"""

train_set = pd.read_csv('milestone1-data/train_set.csv').to_numpy()


val_set = pd.read_csv('milestone1-data/val_set.csv').to_numpy()


test_set = pd.read_csv('milestone1-data/test_set.csv').to_numpy()

#normalisation 

X_train = normalization(train_set)

X_val = normalization(val_set)

X_test = normalization(test_set)

print(len(train_set[0]))
print(train_set[0])
print(len(test_set[0]))

#architechture du réseau de neurones

class NeuralNetwork(nn.Module):
    """1-Layer MNIST classifier"""
    
    def __init__(self):
        super().__init__()
        ### START CODE HERE ###
        self.fc = nn.Linear(784, 10)
        ### END CODE HERE ###

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Flatten to get tensor of shape (batch_size, 784)
        x = x.flatten(start_dim=1)
        ### START CODE HERE ###
        out = self.fc(x)
        return out
        ### END CODE HERE ###

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Predicts classes by calculating the softmax"""
        logits = self.forward(x)
        return F.softmax(logits, dim=1)

model = NeuralNetwork()


loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.05)


# boucle 

def train(model: torch.nn.Module, train_loader: torch.utils.data.DataLoader, loss_fn: torch.nn.Module, optimizer: torch.optim.Optimizer, epochs: int):
    
    # Initialize metrics for loss and accuracy
    loss_metric = metrics.LossMetric()
    acc_metric = metrics.AccuracyMetric(k=1)
    
    # Sets the module in training mode (doesn't have any effect here, but good habit to take)
    model.train()
    
    for epoch in range(1, epochs + 1):
        
        # Progress bar set-up
        pbar = tqdm(total=len(train_loader), leave=True)
        pbar.set_description(f"Epoch {epoch}")
        
        # Iterate through data
        for data, target in train_loader:
            
            ### START CODE HERE ###
            
            # Zero-out the gradients
            optimizer.zero_grad()
            
            # Forward pass
            out = model(data)
            
            # Compute loss
            loss = loss_fn(out, target)
            
            # Backward pass
            loss.backward()
            
            # Optimizer step
            optimizer.step()
            
            ### END CODE HERE ###
            
            # Update metrics & progress bar
            loss_metric.update(loss.item(), data.shape[0])
            acc_metric.update(out, target)
            pbar.update()
            
        # End of epoch, show loss and acc
        pbar.set_postfix_str(f"Train loss: {loss_metric.compute():.3f} | Train acc: {acc_metric.compute() * 100:.2f}%")
        loss_metric.reset()
        acc_metric.reset()



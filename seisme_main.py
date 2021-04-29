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
from tqdm.auto import tqdm
from torch.utils.data import TensorDataset, DataLoader, Dataset

import helper
import metrics
import extract
import math



#extraction des données

train_set = extract.TrainDataset()
val_set = extract.ValDataset()
test_set = extract.TestDataset()

trainlen = len(train_set)


#transformation en DataLoader
batchsize = 32

train_loader = DataLoader(train_set, batch_size=batchsize, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batchsize, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batchsize, shuffle=False)


train_iter = iter(train_loader)
train_data = train_iter.next()
features, labels = train_data


"""
def training_loop(train_set, batchsize, num_epoch):
    
    total_samles= len(train_set)
    n_iterations = math.ceil(total_samples/batchsize)
  """  


"""
train 111 mesures (105 S_a(T), S_a,avg, delta_t1, delta_t2, delta_v, collapse_cap)

val 111 mesures (105 S_a(T), S_a,avg, delta_t1, delta_t2, delta_v, collapse_cap)

test 110 mesures (105 S_a(T), S_a,avg, delta_t1, delta_t2, delta_v)


train_set = pd.read_csv('milestone1-data/train_set.csv').to_numpy()


val_set = pd.read_csv('milestone1-data/val_set.csv').to_numpy()


test_set = pd.read_csv('milestone1-data/test_set.csv').to_numpy()

#normalisation 


train = helper.normalization(train_set)

val = helper.normalization(val_set)

test = helper.normalization(test_set)

X_train = torch.from_numpy(train[:,0:len(train[0])-1])
Y_train = torch.from_numpy(train[:,len(train[0])-1])
train_n_samples = train.shape[0]

train_data_set = TensorDataset(X_train, Y_train)
train_loader = DataLoader(train_data_set)

X, Y = iter(train_loader).next()

print(Y)


X_val = val[:,0:len(val[0])-1].tolist()
Y_val = val[:,len(val[0])-1].tolist()
val_loader = [[X_val[i],[Y_val[i]]] for i in range(len(val))]

X_test = np.array(test[:,0:len(test[0])-1].tolist())

"""



#architechture du réseau de neurones

class ThreeLayerNet(nn.Module):
    """3-Layer neural net"""
    
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(110, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Flatten to get tensor of shape (batch_size, 784)
        x = x.flatten(start_dim=1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        out = self.fc3(x)
        return out

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Predicts classes by calculating the softmax"""
        logits = self.forward(x)
        return F.softmax(logits, dim=1)

# Note: Instance is called three_layer_net instead of model this time around
three_layer_net = ThreeLayerNet()


loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(three_layer_net.parameters(), lr=0.05)



"""

# boucle

num_epoch= 10
total_samples= len(train_set)
n_iterations = math.ceil(total_samples/batchsize)

for epoch in range(num_epoch):
    for i, (inputs, labels) in enumerate(train_loader):
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
        
 

def train(model: torch.nn.Module,
          train_loader: torch.utils.data.DataLoader,
          loss_fn: torch.nn.Module,
          optimizer: torch.optim.Optimizer,
          epochs: int, length: int):
    
    # Initialize metrics for loss and accuracy
    loss_metric = metrics.LossMetric()
    acc_metric = metrics.AccuracyMetric(k=1)
    
    # Sets the module in training mode (doesn't have any effect here, but good habit to take)
    model.train()
    
    for epoch in range(1, epochs + 1):
        
        # Progress bar set-up
        pbar = tqdm(total=length, leave=True)
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


train(three_layer_net, train, loss_fn, optimizer, epochs=10, length=trainlen) """
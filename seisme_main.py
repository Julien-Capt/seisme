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


#extraction des données

"""
train 111 mesures (105 S_a(T), S_a,avg, delta_t1, delta_t2, delta_v, collapse_cap)

val 111 mesures (105 S_a(T), S_a,avg, delta_t1, delta_t2, delta_v, collapse_cap)

test 110 mesures (105 S_a(T), S_a,avg, delta_t1, delta_t2, delta_v)

"""

train_set = pd.read_csv('milestone1-data/train_set.csv').to_numpy()


val_set = pd.read_csv('milestone1-data/val_set.csv').to_numpy()


test_set = pd.read_csv('milestone1-data/test_set.csv').to_numpy()

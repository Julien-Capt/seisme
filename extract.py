import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import math


class TrainDataset(Dataset):
    
    def __init__(self):
        #data loading
        xy = pd.read_csv('milestone1-data/train_set.csv').to_numpy()
        self.x = torch.from_numpy(xy[:, :len(xy[0])-1])
        self.y = torch.from_numpy(xy[:, [len(xy[0])-1]])
        self.n_samples = xy.shape[0]
        
    def __getitem__(self, index):
        # dataset[0]
        return self.x[index], self.y[index]
        
    def __len__(self):
        # len(dataset)
        return self.n_samples
    
    
    
    
class ValDataset(Dataset):
    
    def __init__(self):
        #data loading
        xy = pd.read_csv('milestone1-data/val_set.csv').to_numpy()
        self.x = torch.from_numpy(xy[:, :len(xy[0])-1])
        self.y = torch.from_numpy(xy[:, [len(xy[0])-1]])
        self.n_samples = xy.shape[0]
        
    def __getitem__(self, index):
        # dataset[0]
        return self.x[index], self.y[index]
        
    def __len__(self):
        # len(dataset)
        return self.n_samples
    
    
    

class TestDataset(Dataset):
    
    def __init__(self):
        #data loading
        x = pd.read_csv('milestone1-data/test_set.csv').to_numpy()
        self.x = torch.from_numpy(x[:, :len(x[0])])
        self.n_samples = x.shape[0]
        
    def __getitem__(self, index):
        # dataset[0]
        return self.x[index], self.y[index]
        
    def __len__(self):
        # len(dataset)
        return self.n_samples
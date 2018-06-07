import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pdb
import pandas as pd

class NPZDataset(Dataset):
    def __init__(self, npz_file, normalize, test, mean=None, std=None, as_int=True, mins=None, maxes=None, inter=None, divisions=None):
        self.dict = np.load(npz_file)
        self.normalize = normalize
        self.mean = mean
        self.std  = std
        self.mins = mins
        self.maxes= maxes
        self.inter= inter
        self.X = self.dict['X'].astype(np.float32)
        if as_int:
            self.Y = (self.dict['Y']).astype(int)
        else:
            self.Y = (self.dict['Y']).astype(np.float32)
        if normalize == 1:
            if not test:
                self.mean = np.mean(self.X,axis=0)
                self.std  = np.std(self.X,axis=0)
            self.X = self.X - self.mean
            self.X = self.X / self.std
        elif normalize == 2:
            if not test:
                self.mins = np.min(self.X,axis=0)
                self.maxes= np.max(self.X,axis=0)
                self.inter= (self.maxes-self.mins)/float(divisions)
            #else:
            #    pdb.set_trace()
            newX = np.zeros((self.X.shape[0],self.X.shape[1]*divisions))
            inds = np.arange(0,self.X.shape[1]*divisions,divisions)
            for i in range(divisions):
                newX[:,inds+i] = (self.X >= self.mins + self.inter*i) & (self.X <= self.mins + self.inter*(i+1))
            if test:
                newX[:,inds+divisions-1] = (self.X > self.maxes) | (newX[:,inds+divisions-1]).astype(int)
                newX[:,inds] = (self.X < self.mins) | (newX[:,inds]).astype(int)
            self.X = newX
        elif normalize == 3:
            # first 8 features are non categorical
            #(Pdb) np.setdiff1d(np.arange(len(self.mins)),np.where(np.logical_and(self.mins==0,self.maxes==1))[0])
            #array([0, 1, 2, 3, 4, 5, 6, 7])
            realX = self.X[:,0:8]
            if not test:
                self.mins = np.min(realX,axis=0)
                self.maxes= np.max(realX,axis=0)
                self.inter= (self.maxes-self.mins)/float(divisions)
            #else:
            #    pdb.set_trace()
            newX = np.zeros((realX.shape[0],realX.shape[1]*divisions))
            inds = np.arange(0,realX.shape[1]*divisions,divisions)
            for i in range(divisions):
                newX[:,inds+i] = (realX >= self.mins + self.inter*i) & (realX <= self.mins + self.inter*(i+1))
            if test:
                newX[:,inds+divisions-1] = (realX > self.maxes) | (newX[:,inds+divisions-1]).astype(int)
                newX[:,inds] = (realX < self.mins) | (newX[:,inds]).astype(int)
            self.X = np.concatenate((newX,self.X[:,8:]),axis=1)
            ##if not test:
            ##    dim = self.X.shape[1]
            ##    self.mean = np.mean(self.X,axis=0)
            ##    self.std = np.std(self.X,axis=0)
            ##    for i in range(dim):
            ##        if len(np.unique(self.X[:,i])) > 2:
            ##            self.mean[i] = 0
            ##            self.std[i]  = 1
            ##self.X = self.X - self.mean
            ##self.X = self.X / self.std


    def __len__(self):
        return self.X.shape[0]
    def __getitem__(self, idx):
        X = self.X[idx,:]
        Y = self.Y[idx]
        return X, Y

class NPZDataset2D(Dataset):
    def __init__(self, npz_file, normalize, test, mean=None, std=None, as_int=True):
        self.dict = np.load(npz_file)
        self.normalize = normalize
        self.mean = mean
        self.std  = std
        self.X = self.dict['X'].astype(np.float32)
        if as_int:
            self.Y = (self.dict['Y']).astype(int)
        else:
            self.Y = (self.dict['Y']).astype(np.float32)
        if normalize:
            if not test:
                self.mean = self.X.mean()
                self.std  = self.X.std()
            self.X = self.X - self.mean
            self.X = self.X / self.std

    def __len__(self):
        return self.X.shape[0]
    def __getitem__(self, idx):
        X = self.X[idx,:]
        Y = self.Y[idx]
        return X, Y

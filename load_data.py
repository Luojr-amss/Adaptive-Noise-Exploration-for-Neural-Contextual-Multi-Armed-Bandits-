from sklearn.datasets import fetch_openml
from sklearn.utils import shuffle
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import normalize
import numpy as np
import pandas as pd 

import torch
import torchvision
from torchvision import datasets, transforms


class load_yelp:
    def __init__(self):
        # Fetch data
        self.m = np.load("./yelp_2000users_10000items_entry.npy")
        self.U = np.load("./yelp_2000users_10000items_features.npy")
        self.I = np.load("./yelp_10000items_2000users_features.npy")
        self.n_arm = 10
        self.dim = 20
        self.pos_index = []
        self.neg_index = []
        for i in self.m:
            if i[2] == 1:
                self.pos_index.append((i[0], i[1]))
            else:
                self.neg_index.append((i[0], i[1]))

        self.p_d = len(self.pos_index)
        self.n_d = len(self.neg_index)
        print(self.p_d, self.n_d)
        self.pos_index = np.array(self.pos_index)
        self.neg_index = np.array(self.neg_index)

    def step(self):
        arm = np.random.choice(range(10))
        # print(pos_index.shape)
        pos = self.pos_index[np.random.choice(range(self.p_d), 9, replace=False)]
        neg = self.neg_index[np.random.choice(range(self.n_d), replace=False)]
        X_ind = np.concatenate((pos[:arm], [neg], pos[arm:]), axis=0)
        X = []
        for ind in X_ind:
            # X.append(np.sqrt(np.multiply(self.I[ind], u_fea)))
            X.append(np.concatenate((self.U[ind[0]], self.I[ind[1]])))
        rwd = np.zeros(self.n_arm)
        rwd[arm] = 1
        return np.array(X), rwd
    
    
    
class load_movielen:
    def __init__(self):
        # Fetch data
        self.m = np.load("./movie_2000users_10000items_entry.npy")
        self.U = np.load("./movie_2000users_10000items_features.npy")
        self.I = np.load("./movie_10000items_2000users_features.npy")
        self.n_arm = 10
        self.dim = 20
        self.pos_index = []
        self.neg_index = []
        for i in self.m:
            if i[2] ==1:
                self.pos_index.append((i[0], i[1]))
            else:
                self.neg_index.append((i[0], i[1]))   
            
        self.p_d = len(self.pos_index)
        self.n_d = len(self.neg_index)
        print(self.p_d, self.n_d)
        self.pos_index = np.array(self.pos_index)
        self.neg_index = np.array(self.neg_index)


    def step(self):        
        arm = np.random.choice(range(10))
        #print(pos_index.shape)
        pos = self.pos_index[np.random.choice(range(self.p_d), 9, replace=False)]
        neg = self.neg_index[np.random.choice(range(self.n_d), replace=False)]
        X_ind = np.concatenate((pos[:arm], [neg], pos[arm:]), axis=0)
        X = []
        for ind in X_ind:
            #X.append(np.sqrt(np.multiply(self.I[ind], u_fea)))
            X.append(np.concatenate((self.U[ind[0]], self.I[ind[1]])))
        rwd = np.zeros(self.n_arm)
        rwd[arm] = 1
        return np.array(X), rwd
    
    
    
    

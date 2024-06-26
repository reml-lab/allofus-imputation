"""
MICE model with SGD regressor
"""
import numpy as np
from sklearn.linear_model import SGDRegressor

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader

import math
import time
import pandas as pd
import pickle
from datetime import datetime

import sys
sys.path.append("../..")

from utils.train_utils import lower_upper_bound_func, feature_padding
from utils.data_utils import get_hourly_data, get_multiple_pid, pull_file, FILE_CACHE
        

#### MICE models ####
class MiniBatchSGDRegressor(SGDRegressor):
    def __init__(self, store_model, split_idx, num_epochs=50, batch_size=10000, loss='epsilon_insensitive', 
                       penalty='l2', alpha=0.0001, l1_ratio=0.15, fit_intercept=True, 
                       max_iter=1000, tol=0.001, shuffle=True, verbose=0, epsilon=0.01, 
                       random_state=None, learning_rate='invscaling', eta0=0.01, power_t=0.25, 
                       early_stopping=False, validation_fraction=0.1, n_iter_no_change=5, warm_start=False, 
                       average=False):
        
        # note we cannot use the early stopping here since the sklearn will set aside the validation 
        # which might make the dataset different from what we have
        # also, the input is only the training set, we cannot set aside the validation set from it
        super(MiniBatchSGDRegressor, self).__init__(loss=loss, 
                        penalty=penalty, alpha=alpha, l1_ratio=l1_ratio, fit_intercept=fit_intercept, 
                        max_iter=max_iter, tol=tol, shuffle=shuffle, verbose=verbose, epsilon=epsilon, 
                        random_state=random_state, learning_rate=learning_rate, eta0=eta0, power_t=power_t, 
                        early_stopping=early_stopping, validation_fraction=validation_fraction, n_iter_no_change=n_iter_no_change, 
                        warm_start=warm_start, average=average)
        
        self.store_model = store_model  # whether to store the trained weight and bias for each split
        self.split_idx = split_idx 
        self.batch_size = batch_size
        self.num_epochs = num_epochs  # max_iter does not effect partial_fit methods
    
    def fit(self, X, y):
        for epoch in range(self.num_epochs):
            # randomly shuffle the training data
            rdx = np.arange(X.shape[0])
            np.random.shuffle(rdx)
            X, y = X[rdx, :], y[rdx]
            for bid in range(X.shape[0]//self.batch_size+1):
                X_mb = X[bid*self.batch_size : (bid+1)*self.batch_size, :]
                y_mb = y[bid*self.batch_size : (bid+1)*self.batch_size]
                self.partial_fit(X_mb, y_mb)
        # store the weight and bias into a file
        # actually, when call "fit", the "predict" will also called after the entire fit procedure finishes
        # and the coef_ and intercept_ from the fit and the predict are the same. So the model storing procedure
        # is implemented within fit function. Also, we need to use the milisecond level for the filename, otherwise
        # it would results in overwriting. 
        if self.store_model:
            with open(f"{FILE_CACHE}/split{self.split_idx}_{datetime.now().strftime('%m%d%H%M%S%f')}.pkl", "wb") as fout:
                weight_bias = (self.coef_, self.intercept_)
                pickle.dump(weight_bias, fout)

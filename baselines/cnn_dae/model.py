"""
Denoise Convolutional AutoEncoder
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

from utils.train_utils import lower_upper_bound_func


class CNN_DAE(nn.Module):
    def __init__(self):
        super().__init__()
        #### Encoder ####
        # relu
        self.encoder_relu = nn.ReLU()
        # first conv layer
        self.encoder_conv_1 = torch.nn.Conv1d(in_channels=2, out_channels=4, kernel_size=31, stride=2, padding=11)
        self.encoder_bn_1 = torch.nn.BatchNorm1d(num_features=4)
        # second conv layer
        self.encoder_conv_2 = torch.nn.Conv1d(in_channels=4, out_channels=8, kernel_size=20, stride=2, padding=9)
        self.encoder_bn_2 = torch.nn.BatchNorm1d(num_features=8)
        # third conv layer
        self.encoder_conv_3 = torch.nn.Conv1d(in_channels=8, out_channels=16, kernel_size=10, stride=2, padding=4)
        self.encoder_bn_3 = torch.nn.BatchNorm1d(num_features=16)
        
        #### Decoder ####
        # relu
        self.decoder_relu = torch.nn.ReLU()
        # first transpose convolution layer
        self.decoder_tconv_1 = torch.nn.ConvTranspose1d(in_channels=16, out_channels=8, kernel_size=10, stride=2, padding=4)
        self.decoder_bn_1 = torch.nn.BatchNorm1d(num_features=8)
        # second transpose convolution layer
        self.decoder_tconv_2 = torch.nn.ConvTranspose1d(in_channels=8, out_channels=4, kernel_size=20, stride=2, padding=9)
        self.decoder_bn_2 = torch.nn.BatchNorm1d(num_features=4)
        # third transpose convolution layer
        self.decoder_tconv_3 = torch.nn.ConvTranspose1d(in_channels=4, out_channels=2, kernel_size=31, stride=2, padding=11)
        
    def forward(self, x, max_step_rate, step_rate_mean, step_rate_std):
        """
        x: ["step_rate_norm", 
            "heart_rate_norm"], shape: [bs, L, C], corrupted input

        max_step_rate: shape: [bs, 1]
        step_rate_mean: shape: [bs, 1]
        step_rate_std: shape: [bs, 1]
        """
        # encoder 
        x = self.encoder_relu(self.encoder_bn_1(self.encoder_conv_1(x)))
        x = self.encoder_relu(self.encoder_bn_2(self.encoder_conv_2(x)))
        x = self.encoder_relu(self.encoder_bn_3(self.encoder_conv_3(x)))
        
        # decoder
        x = self.decoder_relu(self.decoder_bn_1(self.decoder_tconv_1(x)))
        x = self.decoder_relu(self.decoder_bn_2(self.decoder_tconv_2(x)))
        x = self.decoder_tconv_3(x)
        
        # limit the output range for the normalized step rate
        upper_bound = ((max_step_rate * 1.5 - step_rate_mean) / step_rate_std).repeat(1, x[:,:,0].shape[-1])
        lower_bound = ((0.0 - step_rate_mean) / step_rate_std).repeat(1, x[:,:,0].shape[-1])
        # limit the range of normalized step rate
        x[:,:,0] = lower_upper_bound_func(x[:,:,0], lower_bound, upper_bound)
        
        return x

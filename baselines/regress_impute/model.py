"""
Regression Imputation
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
sys.append("../..")

from utils.train_utils import lower_upper_bound_func, feature_padding
from utils.data_utils import get_hourly_data, get_multiple_pid, pull_file, FILE_CACHE
from baselines.regress_impute.dataset import AllOfUS_dataset


# linear regression model
# input feature: all the normalized step rate (with missing filled as zero) within the context window, 
# all the normalized heart rate (with missing filled as zero) within the context window
# day of the week and hour of the day of the current predicted hour
class Linear_Regression(nn.Module):
    def __init__(self, kernel_size=(9, 15), pad_full_weeks=True, pid_feat=False, viz_mode=False):
        """
        Args:
            - pid_feat: if adding participant indicator (one hot) to the input feature
            - pad_full_weeks: whether pad the weeks or just get the same day of the week from further weeks
                              if it is False, the context window size would be (kh, 15 + (kw-15)//14 * 2)
        """
        # pid_feat: if adding participant indicator (one hot) to the input feature
        
   
        super().__init__()
        # init
        self.ksize = kernel_size
        self.pad_full_weeks = pad_full_weeks
        self.pid_feat = pid_feat
        self.viz_mode = viz_mode
       
        # define the linear layers do the regression
        # input feature: normalized step rate in the context window: kh * kw - 1 (remove the center one)
        # normalized heart rate in the context window: kh * kw - 1
        # day of the week of the current hour: 7
        # hour of the day of the current hour: 24
        # participant one hot: 100
        
        if pad_full_weeks:
            khkw = self.ksize[0] * self.ksize[1]
        else:
            khkw = self.ksize[0] * (15 + (self.ksize[1]-15)//14 * 2)
        
        input_feat_shape = 2 * (khkw - 1) + 7 + 24
        
        if pid_feat:
            input_feat_shape += 100 
        
        self.regressor = nn.Linear(int(input_feat_shape), 1, bias=True)

    def forward(self, x, max_step_rate, step_rate_mean, step_rate_std, pid_ids):
        """
        x: ["step_rate_norm", 
            "mask_comp",
            "Day of Week",
            "Hour of Day",
            "time_axis",
            "heart_rate_norm"], shape: [bs, 6, kh*kw]

        max_step_rate: shape: [bs, 1]
        step_rate_mean: shape: [bs, 1]
        step_rate_std: shape: [bs, 1]
        pid_ids: shape: [bs, 1]
        """
        
        # get normalized step rate and the mask
        # we use clone since we need to change the input in the following
        steprate_feat = x[:,0,:]
        context_mask = x[:,1,:]
        heartrate_feat = x[:,5,:]

        ###### MOST IMPORTANT!!!!!! ######
        context_mask[:, context_mask.shape[-1]//2] = 0 # don't count the to-be-predicted point
        
        # get day of week and hour of day
        dayweek_feat = x[:,2,:]
        hour_feat = x[:,3,:]
        # one hot encoding for these features
        dayweek_onehot = F.one_hot(dayweek_feat.long(), num_classes=7) # [bs, kh*kw, 7]
        hour_onehot = F.one_hot(hour_feat.long(), num_classes=24) # [bs, kh*kw, 24]
        # concatenate them together
        feat_onehot = torch.cat([dayweek_onehot, hour_onehot], dim=-1)  # [bs, kh*kw, 31]
        
        ############ concatenate it with normalized step rate ########################
        ############ concatenate it with normalized heart rate #######################
        steprate_feat = steprate_feat.unsqueeze(-1)  # [bs, kh*kw, 1]
        heartrate_feat = heartrate_feat.unsqueeze(-1)  # [bs, kh*kw, 1]
        
        ######## IMPOTANT !!!! ###########
        # we need to set the missing step rate or those belonged to other sets as 0.0 (indicates the missing)
        steprate_feat[context_mask.unsqueeze(-1)==0] = 0 
        # we also need to do the same process for the heartrate
        # since step rate and heart rate are correlated (one missing, the other must be missing)
        heartrate_feat[context_mask.unsqueeze(-1)==0] = 0
        
        # for feat_onehot, we only get the center hourly block
        feat_onehot = feat_onehot[:, feat_onehot.shape[1]//2, :]  # [bs, 31]
        # for steprate_feat and heartrate_feat, we get all the context hourly block 
        # except for the center hourly block
        # selected index
        select_index = list(range(steprate_feat.shape[1]))
        select_index.remove(steprate_feat.shape[1]//2)
        steprate_feat = steprate_feat[:, select_index, :].squeeze(-1)  # [bs, (kh*kw-1)]
        heartrate_feat = heartrate_feat[:, select_index, :].squeeze(-1)  # [bs, (kh*kw-1)]
        # concatenate them together
        regressor_feat = torch.cat([steprate_feat, heartrate_feat, feat_onehot.float()], dim=-1)  # [bs, 299]

        # for participant indicator
        if self.pid_feat:
            pid_onehot = F.one_hot(pid_ids.long(), num_classes=100) # [bs, 1, 100]
            pid_onehot = pid_onehot.squeeze(1) # [bs, 100]
            regressor_feat = torch.cat([regressor_feat, pid_onehot.float()], dim=-1)  # [bs, 399]

        feat_avg = self.regressor(regressor_feat)  # [bs, 1]
        # train on unnormalized step rate level
        feat_avg = feat_avg * step_rate_std + step_rate_mean
        # limit the prediction to be the range of 0.0 to 1.5 * max_step_rate
        feat_avg[feat_avg < 0.0] = 0.0
        upper_mask = (feat_avg > 1.5 * max_step_rate)
        feat_avg[upper_mask] = 1.5 * max_step_rate[upper_mask]

        if self.viz_mode:
            return feat_avg, self.regressor.weight
        else:
            return feat_avg  # [bs, 1]

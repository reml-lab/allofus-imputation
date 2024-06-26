"""
Temporally Multi-Scale Sparse Self-Attention Model
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

from utils.train_utils import lower_upper_bound_func, feature_padding
from utils.data_utils import get_hourly_data, get_multiple_pid, pull_file, FILE_CACHE

        
class SelfAttnOneLayerLAPR(nn.Module):
    
    def __init__(self, kernel_size=(9,15), stride=1, pad_full_weeks=True, conv_out_channels=1, d_k=2, d_v=2, if_regress=False, dp_rate=None, pid_feat=False,
                       lapr_rep=False, viz_mode=False):
        """
        Args:
            - kernel_size: the context window size (kh, kw)
            - conv_out_channels: output channel from the convolutional layer
            - d_k: the dimension of the representation for key and query
            - d_v: the dimension of the hidden representation for value
            - pad_full_weeks: whether pad the weeks or just get the same day of the week from further weeks
                              if it is False, the context window size would be (kh, 15 + (kw-15)//14 * 2)
            - if_regress: if use regression, the model operates on the hidden states; otherwise, on the step rates
            - kernel_num: how many kernels do we have in our model
            - dp_rate: dropout rate
            - pid_feat: if adding the participant indicator as the feature
            - lapr_rep: if using normalized step rate to replace the day of the week and hour of the day encoding
            - viz_mode: if True, we return the attention scores
        """
        super().__init__()
        # init
        self.ksize = kernel_size
        self.s = stride
        self.pad_full_weeks = pad_full_weeks
        # self.conv_out_channels = conv_out_channels
        self.conv_out_channels = 1
        self.d_k = d_k
        self.d_v = d_v
        self.if_regress = if_regress
        self.dp_rate = dp_rate
        self.pid_feat = pid_feat
        self.lapr_rep = lapr_rep
        self.viz_mode=viz_mode
        
        # dropout for the regularization
        if self.dp_rate is not None:
            self.dropout = nn.Dropout(p=dp_rate)
        
        # define the layers to process the lapr feature
        # note we set bias=False for all conv layers since layernorm will subtract it out
        # which makes the bias useless
        # for key
        self.conv_key = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=49, stride=1, padding=24, bias=False)
        self.ln_key = nn.LayerNorm(normalized_shape=145)
        # for query
        self.conv_query = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=49, stride=1, padding=24, bias=False)
        self.ln_query = nn.LayerNorm(normalized_shape=145)
        # for value 
        self.conv_value = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=49, stride=1, padding=24, bias=False)
        self.ln_value = nn.LayerNorm(normalized_shape=145)
        # shared for key, query and value 
        self.pool = nn.AvgPool1d(kernel_size=7, stride=6)
        self.relu = nn.ReLU()
        
        # define the linear layers to embed query, key and value
        if self.lapr_rep:
            self.linear_key = nn.Linear(in_features=25, out_features=self.d_k)
            self.linear_query = nn.Linear(in_features=25, out_features=self.d_k)
            if self.if_regress:
                self.linear_value = nn.Linear(in_features=26, out_features=d_v)
                self.regressor = nn.Linear(d_v, 1, bias=True) 
            else:
                self.linear_value = nn.Linear(in_features=26, out_features=1)
        else:
            self.linear_key = nn.Linear(in_features=(25+31), out_features=self.d_k)
            self.linear_query = nn.Linear(in_features=(25+31), out_features=self.d_k)
            if self.if_regress:
                self.linear_value = nn.Linear(in_features=(26+31), out_features=d_v)
                self.regressor = nn.Linear(d_v, 1, bias=True) 
            else:
                self.linear_value = nn.Linear(in_features=(26+31), out_features=1)
        
        # define the parameters for the relative embedding 
        # note we need to get the true context window size depending on pad_full_weeks
        if pad_full_weeks:
            self.ctw_size = self.ksize[0] * self.ksize[1]
        else:
            self.ctw_size = self.ksize[0] * (15 + (self.ksize[1]-15)//14*2)
        self.rel_embed = nn.Parameter(1e-3 * torch.ones(int(self.ctw_size)).float())
        
    def weight_dropout(self, weight):
        """
        Dropout the weights before the softmax.
        Args:
            - weight: the weight we need to dropout
            - dp_rate: dropout rate
        """
        # only do the dropout during the training time
        if self.training:
            p = self.dp_rate
            retain_mask = torch.distributions.Bernoulli(probs=(1-p)).sample(weight.shape).to(weight.device)
            # Note the order of the following two lines!!!
            weight = weight * retain_mask * 1/(1-p)
            weight[~retain_mask.bool()] = -1e9
            return weight
        else:    
            return weight

    def forward(self, x, lapr_feat, max_step_rate, step_rate_mean, step_rate_std, pid_ids):
        """
        x: ["step_rate_norm", 
            "mask_comp",
            "Day of Week",
            "Hour of Day",
            "heart_rate_norm"], shape: [bs, 6, kh*kw]
            
        lapr_feat: shape: [bs, kh*kw, 2*ctw_len+1]
 
        max_step_rate: shape: [bs, 1]
        step_rate_mean: shape: [bs, 1]
        step_rate_std: shape: [bs, 1]
        """
        
        # get shapes for the reshape at the end
        bs, nc, _ = x.shape
        kh, kw = self.ksize
        
        # get normalized step rate and the mask
        # we use clone since we need to change the input in the following
        steprate_feat = x[:,0,:]
        context_mask = x[:,1,:]
        heartrate_feat = x[:,4,:]  # note the dim index is changed!

        ###### MOST IMPORTANT!!!!!! ######
        context_mask[:, context_mask.shape[-1]//2] = 0 # don't count the to-be-predicted point
        
        ############ concatenate it with normalized step rate ########################
        ############ concatenate it with normalized heart rate #######################
        steprate_feat = steprate_feat.unsqueeze(-1)
        heartrate_feat = heartrate_feat.unsqueeze(-1)
        
        ######## IMPOTANT !!!! ###########
        # we need to set the missing step rate or those belonged to other sets as 0.0 (indicates the missing)
        steprate_feat[context_mask.unsqueeze(-1)==0] = 0
        # we also need to do the same process for the heartrate
        # since step rate and heart rate are correlated (one missing, the other must be missing)
        heartrate_feat[context_mask.unsqueeze(-1)==0] = 0
        
        ### key and value ###
        # combine the first and second demension
        bs, khkw, lapr_len = lapr_feat.shape
        lapr_feat_kv = lapr_feat.reshape(-1, lapr_len)
        # add the channel dimension
        lapr_feat_kv = lapr_feat_kv.unsqueeze(1)  # [bs*kh*kw, 1, 145]
        ## key ##
        lapr_info_k = self.ln_key(self.conv_key(lapr_feat_kv)) # [bs*kh*kw, 1, 24]
        lapr_info_k = self.relu(lapr_info_k)
        lapr_info_k = self.pool(lapr_info_k)
        # reshape it back
        lapr_info_k = lapr_info_k.squeeze(1).reshape(bs, khkw, -1) # [bs, kh*kw, 24]
        # concatenate with the step_rate
        feat_k = torch.cat([lapr_info_k, steprate_feat], dim=-1)
        ## value ## 
        lapr_info_v = self.ln_value(self.conv_value(lapr_feat_kv)) # [4566645, 1, 24]
        lapr_info_v = self.relu(lapr_info_v)
        lapr_info_v = self.pool(lapr_info_v)
        # reshape it back
        lapr_info_v = lapr_info_v.squeeze(1).reshape(bs, khkw, -1) # [33827, 135, 24]
        # concatenate with the step_rate
        feat_v = torch.cat([lapr_info_v, steprate_feat, heartrate_feat], dim=-1)
        
        ### query ###
        # combine the first and second demension
        lapr_feat_q = lapr_feat[:, khkw//2, :]  # [33827, 145]
        # add the channel dimension
        lapr_feat_q = lapr_feat_q.unsqueeze(1)  # [33827, 1, 145]
        lapr_info_q = self.ln_query(self.conv_query(lapr_feat_q)) 
        lapr_info_q = self.relu(lapr_info_q)
        lapr_info_q = self.pool(lapr_info_q) # [33827, 1, 24]
        # concatenate with the step_rate
        feat_q = torch.cat([lapr_info_q, steprate_feat[:, steprate_feat.shape[1]//2, :].unsqueeze(1)], dim=-1)
        
        if not self.lapr_rep:
            # get day of week and hour of day
            dayweek_feat = x[:,2,:]
            hour_feat = x[:,3,:]
            # one hot encoding for these features
            dayweek_onehot = F.one_hot(dayweek_feat.long(), num_classes=7)
            hour_onehot = F.one_hot(hour_feat.long(), num_classes=24)
            # concatenate them together
            feat_onehot = torch.cat([dayweek_onehot, hour_onehot], dim=-1).float()  # [bs, kh*kw, 31]
            feat_k = torch.cat([feat_onehot, feat_k], dim=-1)
            feat_q = torch.cat([feat_onehot[:, feat_onehot.shape[1]//2, :].unsqueeze(1), feat_q], dim=-1)
            feat_v = torch.cat([feat_onehot, feat_v], dim=-1)
        
        # value
        value = self.linear_value(feat_v)  # [bs, kh*kw, 1]
        # key
        key = self.linear_key(feat_k)  # [bs, kh*kw, d_k]
        # query
        query = self.linear_query(feat_q)  # [bs, 1, d_k]
               
        ### we add the relative position embedding to qk directly
        scores = (torch.matmul(query, key.transpose(-1,-2)) + self.rel_embed) / math.sqrt(self.d_k)  # [bs, 1, kh*kw]
        scores = scores.masked_fill(context_mask.unsqueeze(1)==0, -1e9)
        # apply the dropout on the logits before the softmax
        if self.dp_rate is not None:
            scores = self.weight_dropout(scores)
        # get the attention score
        p_attn = F.softmax(scores, dim=-1) # [bs, 1, kh*kw]
        # compute the weighted average
        feat_avg = torch.matmul(p_attn, value)  # [bs, 1, d_v] or [bs, 1, 1]
        # apply the regressor
        if self.if_regress:
            feat_avg = self.regressor(feat_avg)
        # deal with the shape
        feat_avg = feat_avg.squeeze(-1) # [bs, 1]
        # train on unnormalized step rate level
        feat_avg = feat_avg * step_rate_std + step_rate_mean
        # limit the prediction to be the range of 0.0 to 1.5 * max_step_rate
        feat_avg[feat_avg < 0.0] = 0.0
        upper_mask = (feat_avg > 1.5 * max_step_rate)
        feat_avg[upper_mask] = 1.5 * max_step_rate[upper_mask]

        if self.viz_mode:
            return feat_avg, p_attn
        else:
            return feat_avg  # [bs, 1]
        
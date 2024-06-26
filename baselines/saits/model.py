"""
This script is to define models with multiple layers
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
from baselines.saits.dataset import AllOfUsDatasetLAPR, BatchCollate
from model import SelfAttnOneLayerLAPR


class SelfAttnMultiLayerLAPR(SelfAttnOneLayerLAPR):
    """
    Two layer model with local context window and local activity profile representations
    """
    def __init__(self, *args, **kwargs):
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
        super(SelfAttnOneLayerLAPR, self).__init__(*args, **kwargs)
        # define the module for the second layer 
        # Note the second layer does not have the local activity profile representation anymore
        # Also note that if_regress in the argument should always be True in this case
        self.linear_key_second = nn.Linear(in_features=self.d_v, out_features=self.d_k)
        self.linear_query_second = nn.Linear(in_features=self.d_v, out_features=self.d_k)
        self.linear_value_second = nn.Linear(in_features=self.d_v, out_features=1)
        self.rel_embed_second = nn.Parameter(1e-3 * torch.ones(int(self.ctw_size)).float())
        
        #self.register_buffer('q_mask', torch.ones(self.ctw_size), persistent=False)

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
        # -----------
        # First Layer
        # -----------

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
        steprate_feat = steprate_feat.unsqueeze(-1)  # [bs, kh*kw, 1]
        heartrate_feat = heartrate_feat.unsqueeze(-1)  # [bs, kh*kw, 1]
        
        ######## IMPOTANT !!!! ###########
        # we need to set the missing step rate or those belonged to other sets as 0.0 (indicates the missing)
        steprate_feat[context_mask.unsqueeze(-1)==0] = 0
        # we also need to do the same process for the heartrate
        # since step rate and heart rate are correlated (one missing, the other must be missing)
        heartrate_feat[context_mask.unsqueeze(-1)==0] = 0
        
        ### key and value ###
        # combine the first and second demension
        bs, khkw, lapr_len = lapr_feat.shape
        lapr_feat = lapr_feat.reshape(-1, lapr_len)
        # add the channel dimension
        lapr_feat = lapr_feat.unsqueeze(1)  # [bs*kh*kw, 1, 145]
        ## key ##
        lapr_info_k = self.ln_key(self.conv_key(lapr_feat)) # [bs*kh*kw, 1, 24]
        lapr_info_k = self.relu(lapr_info_k)
        lapr_info_k = self.pool(lapr_info_k)
        # reshape it back
        lapr_info_k = lapr_info_k.squeeze(1).reshape(bs, khkw, -1) # [bs, kh*kw, 24]
        # concatenate with the step_rate
        feat_k = torch.cat([lapr_info_k, steprate_feat], dim=-1)
        ## value ## 
        lapr_info_v = self.ln_value(self.conv_value(lapr_feat)) # [bs*kh*kw, 1, 24]
        lapr_info_v = self.relu(lapr_info_v)
        lapr_info_v = self.pool(lapr_info_v)
        # reshape it back
        lapr_info_v = lapr_info_v.squeeze(1).reshape(bs, khkw, -1) # [bs, kh*kw, 24]
        # concatenate with the step_rate
        feat_v = torch.cat([lapr_info_v, steprate_feat, heartrate_feat], dim=-1)
        
        ### query ###
        ## this time, we use lapr_feat_kv as lapr_feat_q, since we don't only use the center hourly block anymore
        lapr_info_q = self.ln_query(self.conv_query(lapr_feat)) 
        lapr_info_q = self.relu(lapr_info_q)
        lapr_info_q = self.pool(lapr_info_q) # [bs*kh*kw, 1, 24]
        # reshape it back
        lapr_info_q = lapr_info_q.squeeze(1).reshape(bs, khkw, -1)  # [bs, kh*kw, 24]
        # concatenate with the step_rate
        feat_q = torch.cat([lapr_info_q, steprate_feat], dim=-1)  # [bs, kh*kw, 25]
        
        if not self.lapr_rep:
            # get day of week and hour of day
            dayweek_feat = x[:,2,:]
            hour_feat = x[:,3,:]
            # one hot encoding for these features
            dayweek_onehot = F.one_hot(dayweek_feat.long(), num_classes=7)
            hour_onehot = F.one_hot(hour_feat.long(), num_classes=24)
            # concatenate them together
            feat_onehot = torch.cat([dayweek_onehot, hour_onehot], dim=-1).float()  # [bs, kh*kw, 31]
            feat_k = torch.cat([feat_onehot, feat_k], dim=-1)  # [bs, kh*kw, 31+24+1]
            feat_q = torch.cat([feat_onehot, feat_q], dim=-1)  # [bs, kh*kw, 31+24+1]
            feat_v = torch.cat([feat_onehot, feat_v], dim=-1)  # [bs, kh*kw, 31+24+2]
        
        # value
        value = self.linear_value(feat_v)  # [bs, kh*kw, d_v]
        # key
        key = self.linear_key(feat_k)  # [bs, kh*kw, d_k]
        # query
        query = self.linear_query(feat_q)  # [bs, kh*kw, d_k]
               
        ### we add the relative position embedding to qk directly
        #scores = (torch.matmul(query, key.transpose(-1,-2)) + self.rel_embed) / math.sqrt(self.d_k)  # [bs, kh*kw, kh*kw]
        scores = torch.matmul(query, key.transpose(-1,-2)) / math.sqrt(self.d_k)  # [bs, kh*kw, kh*kw]
        # get the attention mask which has the shape khkw * khkw
        # k_mask = context_mask  # [bs, kh*kw]
        q_mask = torch.ones_like(context_mask, device=context_mask.device) # [bs, kh*kw]
        attn_mask = torch.matmul(q_mask.unsqueeze(-1), context_mask.unsqueeze(1))  # [bs, kh*kw, 1] @ [bs, 1, kh*kw]  --> [bs, khkw, khkw]
        scores = scores.masked_fill(attn_mask==0, -1e9)
        #scores = scores.masked_fill(context_mask.unsqueeze(1)==0, -1e9)   # context_mask shape: [bs, kh*kw, 1]
        # apply the dropout on the logits before the softmax
        if self.dp_rate is not None:
            scores = self.weight_dropout(scores)
        # get the attention score
        p_attn = F.softmax(scores, dim=-1) # [bs, kh*kw, kh*kw]
        # compute the weighted average
        feat_avg = torch.matmul(p_attn, value)  # [bs, kh*kw, kh*kw] @ [bs, kh*kw, d_v] --> [bs, kh*kw, d_v]
        # apply the regressor
        #if self.if_regress:
        #feat_avg = self.regressor(feat_avg)
        
        # ------------
        # Second layer
        # ------------
        key = self.linear_key_second(feat_avg)  # [bs, kh*kw, d_k]
        value = self.linear_value_second(feat_avg)  # [bs, kh*kw, 1]
        query = self.linear_query_second(feat_avg[:, feat_avg.shape[1]//2, :].unsqueeze(1))  # [bs, 1, d_k]
        scores = torch.matmul(query, key.transpose(-1,-2)) / math.sqrt(self.d_k)  # [bs, 1, kh*kw]
        p_attn = F.softmax(scores, dim=-1) # [bs, 1, kh*kw]
        # compute the weighted average
        feat_avg = torch.matmul(p_attn, value)  # [bs, 1, 1]

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


if __name__ == "__main__":
    device = torch.device(f'cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"device: {device}")

    # get the data for the participant
    pull_file("pid_data.pkl")
    with open(f"{FILE_CACHE}/pid_data.pkl", "rb") as fin:
        pid_data = pickle.load(fin)
    
    # get the participant id list
    pull_file("df_cohort_top100.parquet")  # created by get_cohort_aaai.ipynb
    df_cohort = pd.read_parquet(f"{FILE_CACHE}/df_cohort_top100.parquet")
    pid_list = df_cohort.index.tolist()
    
    ks = (9, 71)
    ctx_len = 72  # the window size on one side for the lapr feature
    split_idx = 0
    batch_size=5
    pad_full_weeks = False

    # define the dataset and dataloader
    batch_collate = BatchCollate(ctx_len)
    ## test ##
    test_dataset = AllOfUsDatasetLAPR(pid_list, pid_data, split_idx, dataset="test", ks=ks, pad_full_weeks=pad_full_weeks)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, collate_fn=batch_collate, pin_memory=False)
    print(f"split {split_idx} | test | input_feat shape: {test_dataset.input_feat_pids.shape} | lapr_feat shape: {test_dataset.lapr_pids.shape}")

    model = SelfAttnMultiLayerLAPR( kernel_size=ks, 
                                    stride=1, 
                                    pad_full_weeks=pad_full_weeks, 
                                    conv_out_channels=1,
                                    d_k=8, 
                                    d_v=32, 
                                    if_regress=True, 
                                    dp_rate=None, 
                                    pid_feat=False, 
                                    lapr_rep=False).to(device)
    
    for idx, (input_feat, lapr_feat, step_gt, max_sr, sr_mean, sr_std, pid_ids) in enumerate(test_loader):
        if idx > 0:
            break

        input_feat = input_feat.to(device)
        lapr_feat = lapr_feat.to(device)
        step_gt = step_gt.to(device)
        max_sr = max_sr.to(device)
        sr_mean = sr_mean.to(device)
        sr_std = sr_std.to(device) 
        pid_ids = pid_ids.to(device)
    
        print(input_feat.shape, lapr_feat.shape, step_gt.shape, max_sr.shape, sr_mean.shape, sr_std.shape, pid_ids.shape)
        
        start_time = time.time()
        output = model(input_feat, lapr_feat, max_sr, sr_mean, sr_std, pid_ids) # forward pass shape: [bs, 1]  
        print(f"forward pass time: {(time.time() - start_time):.4f} seconds")
        print(f"output shape: {output.shape}")    
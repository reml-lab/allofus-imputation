import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import pickle
import copy
import math
import re
import datetime

import warnings
warnings.filterwarnings("ignore")
np.set_printoptions(suppress=True)


def new_dir(folder_path):
    """
    construct a new folder
    """
    if os.path.exists(folder_path):
        print(f"{folder_path} has already existed!")
    else:
        os.makedirs(folder_path)
        print(f"{folder_path} created!")


def count_parameters(model):
    """
    count the trainable parameters in the model
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def mse_loss(output, true, mask=None, norm=True):
    # micro mse loss
    loss = (output - true) ** 2
    if mask is not None:
        loss = loss * mask
        if norm:
            if mask.sum() == 0:
                return loss.sum()
            else:
                return loss.sum() / mask.sum()
        else:
            return loss.sum()
    else:
        if norm:
            return loss.mean()
        else:
            return loss.sum()


def mae_loss(output, true, mask=None, norm=True):
    # micro mse loss
    loss = torch.abs(output - true)
    if mask is not None:
        loss = loss * mask
        if norm:
            if mask.sum() == 0:
                return loss.sum()
            else:
                return loss.sum() / mask.sum()
        else:
            return loss.sum()
    else:
        if norm:
            return loss.mean()
        else:
            return loss.sum()


def lower_upper_bound_func(tensor, lower_bound, upper_bound):
    """
    The function to limit values in the tensor to be between lower_bound and upper_bound
    """
    lower_mask = (tensor < lower_bound)
    tensor[lower_mask] = lower_bound[lower_mask]
    upper_mask = (tensor > 1.5 * upper_bound)
    tensor[upper_mask] = 1.5 * upper_bound[upper_mask]
    
    return tensor


def feature_padding(features:torch.tensor, kernel_size=(9, 15), device="cpu"):
    """
    Pad features, so that we can unfold into the same size context windows for the conv based models
    Args:
        - features: [bs, num_feature, num_hour, num_study_day], the input_feature we need to pad on
        - kernel_size: (kh, kw), the hight and width of the context window (minimum value for kh and kw is 1)
        - device: the device for training the model (gpu or cpu)
    """
    
    assert isinstance(kernel_size[0], int) and (kernel_size[0]-1)%2==0 and ((kernel_size[0]-1)//2)>=0,\
           "context window has a wrong height!"
    assert isinstance(kernel_size[1], int) and (kernel_size[1]-1)%7==0 and (((kernel_size[1]-1)//7)/2)>=0, \
           "context window has a wrong width!" 
    
    pad_height = (kernel_size[0] - 1) // 2
    pad_width = (kernel_size[1] - 1) //2
    

    if pad_width == 0:
        # even if the pad_width == 0, we need to pad since there is one day shift for the bottom and the top
        # but this additional padding will be removed at the end
        # otherwise, the total number of context windows will be wrong
        pad_left = torch.zeros((features.shape[0], features.shape[1], features.shape[2], 1)).to(device)
        pad_right = torch.zeros((features.shape[0], features.shape[1], features.shape[2], 1)).to(device)
    else:
        # get the valid padding on the left
        pad_left = torch.zeros((features.shape[0], features.shape[1], features.shape[2], pad_width)).to(device)
        # get the valid padding on the right
        pad_right = torch.zeros((features.shape[0], features.shape[1], features.shape[2], pad_width)).to(device)
        
    # check the correctness
    assert pad_left.shape == pad_right.shape, "pad_left and pad_right have different shapes"

    # pad the input feature
    features_pad = torch.concat((pad_left, features, pad_right), dim=-1)
    
    
    if pad_height != 0:
        # get the valid padding on the top 
        pad_top = features[:, :, -pad_height:, :]
        # get the valid padding on the bottom
        pad_bottom = features[:, :, :pad_height, :]
        
        # here, we use the same way of padding for pad_width and pad_width for the simplicity
        if (pad_width==0) or (pad_width==1):
            # set all the paddings before and at the first day as zeros
            pad_top = torch.concat((torch.zeros((pad_top.shape[0], pad_top.shape[1], pad_top.shape[2], 2)).to(device), pad_top), dim=-1)
            ## we don't need to pad at the right side anymore for the top ##
            # set all the paddings at or after the last day as zeros
            pad_bottom = torch.concat((pad_bottom, torch.zeros((pad_bottom.shape[0], pad_bottom.shape[1], pad_bottom.shape[2], 2)).to(device)), dim=-1)
            ## we don't need to pad at the left side anymore for the bottom ##
            
        else:
            # set all the paddings before and at the first day as zeros
            pad_top = torch.concat((torch.zeros((pad_top.shape[0], pad_top.shape[1], pad_top.shape[2], pad_width+1)).to(device), pad_top), dim=-1)
            # Note that we don't throw away data from any day since all of them are useful
            # set all the paddings after the last day as zeros
            pad_top = torch.concat((pad_top, torch.zeros((pad_top.shape[0], pad_top.shape[1], pad_top.shape[2], pad_width-1)).to(device)), dim=-1)
            # set all the paddings before the first day as zeros
            pad_bottom = torch.concat((torch.zeros((pad_bottom.shape[0], pad_bottom.shape[1], pad_bottom.shape[2], pad_width-1)).to(device), pad_bottom), dim=-1)    
            # set all the paddings at or after the last day as zeros
            pad_bottom = torch.concat((pad_bottom, torch.zeros((pad_bottom.shape[0], pad_bottom.shape[1], pad_bottom.shape[2], pad_width+1)).to(device)), dim=-1)

        # check the correctness
        assert pad_top.shape == pad_bottom.shape, "pad_top and pad_bottom have different shapes"
        
        features_pad = torch.concat((pad_top, features_pad, pad_bottom), dim=2)
        
    ##  if pad_height == 0, then there is no need to pad on top and bottom ##

    # correctness check
    if pad_height != 0:
        if (pad_width==0) or (pad_width==1):
            assert (features_pad[:, :, :pad_height, 2:]==features[:, :, -pad_height:, :]).all(), "top pad is weird"
            assert (features_pad[:, :, -pad_height:, :-2]==features[:, :, :pad_height, :]).all(), "bottom pad is weird"
            assert (features_pad[:, :, pad_height:-pad_height, 1:-1]==features[:, :, :, :]).all(), "middel feature is weird"
        else:
            assert (features_pad[:, :, :pad_height, (pad_width+1):-(pad_width-1)]==features[:, :, -pad_height:, :]).all(), "top pad is weird"
            assert (features_pad[:, :, -pad_height:, (pad_width-1):-(pad_width+1)]==features[:, :, :pad_height, :]).all(), "bottom pad is weird"
            assert (features_pad[:, :, pad_height:-pad_height, pad_width:-pad_width]==features[:, :, :, :]).all(), "middle feature is weird"
   
    if pad_width == 0:
        # we need to remove the first and the last two columns
        # in order to keep the number of context window correct
        features_pad = features_pad[:, :, :, 1:-1]
    
    # correctness check    
    assert features_pad.shape[2] == (features.shape[2] + (kernel_size[0]-1)), "padded feature has wrong height"
    assert features_pad.shape[3] == (features.shape[3] + (kernel_size[1]-1)), "padded feature has wrong width"
    
    
    return features_pad


### BRITS ###
def make_forward_backward_data(input_feat, max_sr, sr_mean, sr_std, ks):
    # make the forward and backward input features for BRITS model
    # input_feat: [bs, 6, kh*kw]
    # ["step_rate_norm", "mask_comp", "Day of Week", "Hour of Day", "time_axis", "heart_rate_norm"]
    
    data = {"forward":{}, "backward":{}}  # store all the data including both forward and backward
    
    ### step 1: reorder the feature ###
    input_feat = input_feat.reshape(input_feat.shape[0], input_feat.shape[1], ks[0], -1)  # row major
    input_feat = input_feat.transpose(-1, -2)  # column major
    input_feat = input_feat.reshape(input_feat.shape[0], input_feat.shape[1], -1)  # reorder into time order
    input_feat = input_feat.transpose(-1, -2) # put the time dimension on the second dim
    
    ### step 2: split the features ###
    # normalized step rate, normalized heart rate, computational mask
    nsr_feat = input_feat[:, :, 0].unsqueeze(-1)
    nhr_feat = input_feat[:, :, 5].unsqueeze(-1)
    comp_mask = input_feat[:, :, 1].unsqueeze(-1)
    ###### MOST IMPORTANT!!!!!! ######
    comp_mask[:, comp_mask.shape[1]//2, :] = 0 # don't count the to-be-predicted point
    nsr_feat[comp_mask==0] = 0
    nhr_feat[comp_mask==0] = 0
    # check the correctness for the center hourly block
    assert nsr_feat[:, nsr_feat.shape[1]//2, :].sum().item()==0, "nsr_feat center hour is not masked!"
    assert nhr_feat[:, nhr_feat.shape[1]//2, :].sum().item()==0, "nhr_feat center hour is not masked!"
    # concatenate nsr and nhr
    nsr_nhr_values = torch.concat([nsr_feat, nhr_feat], dim=-1)
    
    # day of the week and hour of the day indicator
    dw_feat = input_feat[:, :, 2]
    hd_feat = input_feat[:, :, 3]
    # one hot encoding for these features
    dw_onehot = F.one_hot(dw_feat.long(), num_classes=7)
    hd_onehot = F.one_hot(hd_feat.long(), num_classes=24)
    # concatenate them together
    dw_hd_values = torch.cat([dw_onehot, hd_onehot], dim=-1)  # [bs, kh*kw, 31]
    
    ### step 3: get the deltas ###
    tp_values = input_feat[:, :, 4]  # get the time axis
    ## get the forward deltas ##
    deltas_forward = torch.zeros_like(tp_values)
    for h in range(1, tp_values.shape[1]):
        deltas_forward[:, h] = (tp_values[:, h] - tp_values[:, h-1]) + (1-comp_mask[:, h-1, :].squeeze(-1)) * deltas_forward[:, h-1]
    ## get the backward deltas ##
    # we then need reverse the mask_test to get the backed mask
    reverse_index = (comp_mask.shape[1]-1) - np.arange(comp_mask.shape[1])
    comp_mask_backward = comp_mask[:, reverse_index, :]
    # s[t-1]-s[t] but with backward order
    tp_values_backward = torch.abs((tp_values - tp_values[:, -1].unsqueeze(-1))[:,reverse_index])
    # get the backward deltas #
    deltas_backward = torch.zeros_like(tp_values_backward)
    for h in range(1, tp_values_backward.shape[1]):
        deltas_backward[:, h] = (tp_values_backward[:, h] - tp_values_backward[:, h-1]) + (1-comp_mask_backward[:, h-1].squeeze(-1)) * deltas_backward[:, h-1]
    
    ### step 4: put all the data into the dictionary ###
    data["forward"]["nsr_nhr_values"] = nsr_nhr_values  # [bs, khkw, 2]
    data["forward"]["dw_hd_values"] = dw_hd_values  # [bs, khkw, 31]
    data["forward"]["masks"] = comp_mask.repeat(1, 1, 2)  # [bs, khkw, 1] note mask for heart rate and step rate are the same!
    # note we need to normalize the delta as well in order to make them into [0,1]
    # which is to ease the optimization
    data["forward"]["deltas"] = deltas_forward.unsqueeze(-1).repeat(1, 1, 2) / (24*ks[1])  # [bs, khkw, 2]
    
    data["backward"]["nsr_nhr_values"] = nsr_nhr_values[:, reverse_index, :]
    data["backward"]["dw_hd_values"] = dw_hd_values[:, reverse_index, :]
    data["backward"]["masks"] = comp_mask_backward.repeat(1, 1, 2)
    # note we need to normalize the delta as well in order to make them into [0,1]
    # which is to ease the optimization
    data["backward"]["deltas"] = deltas_backward.unsqueeze(-1).repeat(1, 1, 2) / (24*ks[1]) # [bs, khkw, 2]
    
    data["max_sr"] = max_sr
    data["sr_mean"] = sr_mean
    data["sr_std"] = sr_std
    
    return data
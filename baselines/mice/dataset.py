"""
This script is to define the dataset class for MICE
"""
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

import copy
from tqdm import tqdm
import time
import pickle

import sys
sys.path.append("../..")
from utils.data_utils import pull_file, FILE_CACHE
from utils.train_utils import feature_padding


class MiceDataset:
    def __init__(self, pid_data, split_idx, dataset, kernel_size, pad_full_weeks):
        self.split_idx = split_idx
        self.dataset = dataset
        self.ks = kernel_size
        self.pad_full_weeks = pad_full_weeks
        self.unfold = nn.Unfold(kernel_size=self.ks, stride=1)
        self.input_feat, self.step_gt, self.max_sr, self.sr_mean, self.sr_std, self.pid_ids = self.process(pid_data)
        
    def process(self, pid_data):
        input_feat_list = []
        step_gt_list = []
        step_rate_mean_list = []
        step_rate_std_list = []
        max_step_rate_list = []
        part_id_list = []
        
        for part_id, (conv_feat, feature_list, step_rate_mean, step_rate_std) in enumerate(tqdm(pid_data)):
            #### get the groundtruth, valid minutes and corresponding masks ####
            step_gt = conv_feat[[feature_list.index("steps"),                                              # 0
                                 feature_list.index("valid_minutes"),                                      # 1
                                 feature_list.index(f"{self.dataset}_mask_split{self.split_idx}")], :, :]   # 2
            # change the shape 
            step_gt = step_gt.reshape(step_gt.shape[0], -1)
            # transpose the shape
            step_gt = step_gt.T
            # get the mask for the dataset (train, valid or test)
            dataset_mask = np.argwhere(step_gt[:, 2]).squeeze(1)
            step_gt = step_gt[dataset_mask, ...]
            assert step_gt[:, 2].sum()==step_gt.shape[0], f"pid {part_id} | split {self.split_idx} | {self.dataset}_step_gt mask is wrong!"
            step_gt_list.append(step_gt)
                                
            #### get participant level mean and maximum step rate ####
            # Note that we need to use the training dataset to get the participant mean
            # and the maximum step rate
            train_step_gt = conv_feat[[feature_list.index("steps"), 
                                       feature_list.index("valid_minutes"), 
                                       feature_list.index(f"train_mask_split{self.split_idx}")], :, :] # [3, h, w]
            train_step_gt = torch.from_numpy(train_step_gt)
    
            train_steps = train_step_gt[0,:,:]
            train_valid_min = train_step_gt[1,:,:]
            train_mask = train_step_gt[2,:,:]
            # compute the maximum step rate on the training dataset of that particular split
            max_step_rate = (train_steps[train_mask==1] / train_valid_min[train_mask==1]).max()
            # repeat it for each context_window
            max_step_rate = max_step_rate.unsqueeze(0).repeat(step_gt.shape[0], 1)
            max_step_rate_list.append(max_step_rate.numpy())

            #### get the step_rate_mean and step_rate_std ####
            # print(step_rate_mean, step_rate_std, part_step_rate_mean.item(), max_step_rate.item())
            # repeat the step_rate_mean and step_rate_std for each context window
            step_rate_mean = torch.tensor([step_rate_mean]).unsqueeze(0).repeat(step_gt.shape[0], 1)
            step_rate_std = torch.tensor([step_rate_std]).unsqueeze(0).repeat(step_gt.shape[0], 1)
            step_rate_mean_list.append(step_rate_mean.numpy())
            step_rate_std_list.append(step_rate_std.numpy())

            #### get the part_id ####
            part_id = torch.tensor([part_id]).unsqueeze(0).repeat(step_gt.shape[0], 1)
            part_id_list.append(part_id.numpy())

            #### get the input features for the model ####
            conv_feat_tensor = torch.from_numpy(conv_feat[None, ...]).float() # [1, 13, h, w]
            # we also concatenate the day of the week and hour of the day of the center hourly block to the 
            # feature set
            input_feat = conv_feat_tensor[:, [feature_list.index("step_rate_norm"), 
                                              feature_list.index("Day of Week"),
                                              feature_list.index("Hour of Day"),
                                              feature_list.index(f"{self.dataset}_mask_comp_split{self.split_idx}")], :, :]
            input_pad_feat_tensor = feature_padding(input_feat, kernel_size=self.ks, device='cpu')# print(input_pad_feat.shape)
            input_feat_unfold = self.make_context_windows(input_pad_feat_tensor)
            input_feat_unfold = input_feat_unfold.squeeze(0).numpy()
            # select those context windows which are belonged to this dataset
            input_feat_unfold = input_feat_unfold[dataset_mask, ...]
            # set the mask of the center hourly block as zero
            # Note that we cannot make the target value in the train as np.nan
            # otherwise, the model cannot be trained for predicting this position
            if self.dataset=="valid" or self.dataset=="test":
                input_feat_unfold[:, -1, input_feat_unfold.shape[2]//2] = 0.0
            # set the corresponding normalized step rate as np.nan for missing places
            feat_nsr = input_feat_unfold[:, 0, :]
            feat_comp_mask = input_feat_unfold[:, -1, :]
            feat_nsr[feat_comp_mask==0] = np.nan
            # reorder feat_nsr for MICE
            kh = self.ks[0]
            kw = feat_nsr.shape[1] // kh
            comp_index = self.get_index_start_end_alter(kh, kw)
            feat_nsr = feat_nsr[:, comp_index]
            if self.dataset=="valid" or self.dataset=="test":
                assert np.isnan(feat_nsr[:, -1]).all(), f"pid {part_id} | split {self.split_idx} | {self.dataset} feat_nsr is wrong!"                    
            elif self.dataset == "train":
                assert not np.isnan(feat_nsr[:, -1]).any(), f"pid {part_id} | split {self.split_idx} | {self.dataset} feat_nsr is wrong!" 
            # also need to add the day of the week and hour of the day feature of the center hourly block
            feat_dw_hr = input_feat_unfold[:, [1,2], input_feat_unfold.shape[-1]//2]
            # construct the one hot vector for the day of the week and hour of the day feature for the center hourly block
            feat_one_hot = self.create_one_hot_dayweek_hour(feat_dw_hr)
            # concatenate the normalized step rate and the day of the week and hour of the day
            # we put the one hot vector of day of the week and hour of the day before all the normalized step rate 
            # feature, in order to make sure day of the week and hour of the day feature can be seen when computing 
            # every hourly block.
            mice_feat = np.concatenate([feat_one_hot, feat_nsr], axis=-1)
            # print(f"input_pad_feat.shape: {input_pad_feat.shape}")
            input_feat_list.append(mice_feat)

        # concatenate them for all participants in the pid_data
        print(f"split {self.split_idx} | dataset {self.dataset} | begin concatenating ...")
        part_id_pids = np.concatenate(part_id_list, axis=0).astype(np.uint8)  # int8
        input_feat_pids = np.concatenate(input_feat_list, axis=0).astype(np.float32)  # float32
        step_gt_pids = np.concatenate(step_gt_list, axis=0).astype(np.float32)
        max_step_rate_pids = np.concatenate(max_step_rate_list, axis=0).astype(np.float32)
        step_rate_mean_pids = np.concatenate(step_rate_mean_list, axis=0).astype(np.float32)
        step_rate_std_pids = np.concatenate(step_rate_std_list, axis=0).astype(np.float32)
        print(f"split {self.split_idx} | dataset {self.dataset} | finish concatenating!")

        return input_feat_pids, step_gt_pids, max_step_rate_pids, step_rate_mean_pids, step_rate_std_pids, part_id_pids
            
    def make_context_windows(self, x):
        """
        use nn.Unfold to get the context windows from input_pad_feat
        """
        bs, nc, h, w = x.shape
        # unfold 
        x = self.unfold(x) # [1, nc * kh * kw, L] middle shape: channel * kh * kw, last: # of patches
        # transpose
        x = x.transpose(1, 2) # [1, L, nc * kh * kw]
        # split into different features
        x = x.view(x.shape[0], x.shape[1], nc, -1) # [1, L, 6, kh * kw]
        # deal with the case that pad_full_week is False
        if not self.pad_full_weeks:
            # get the indeces along kw to keep
            center_idx = self.ks[1] // 2
            num_week = (self.ks[1]-1) / 14
            index_shift = np.array(list(range(1, 7)) + (np.arange(1, num_week+1)*7).tolist()) # note np.array is applied on the entire thing
            right_idx = center_idx + index_shift
            left_idx = center_idx - index_shift
            keep_idx = sorted(left_idx.tolist() + [center_idx] + right_idx.tolist())
            keep_idx = list(map(int, keep_idx))  # convert the data type into int
            # get the corresponding feature
            x = x.view(x.shape[0], x.shape[1], x.shape[2], self.ks[0], self.ks[1])
            x = x[:, :, :, :, keep_idx]
            # conver it back to original shape
            x = x.contiguous().view(x.shape[0], x.shape[1], x.shape[2], -1)  # [1, L, 6, kh*len(keep_idx)]
        # change the dtype for the nn.Linear
        x = x.float()

        return x
                                
    def create_one_hot_dayweek_hour(self, feat_dw_hr):
        feat_dw_hr = feat_dw_hr.astype("int")
        feat_dw = feat_dw_hr[:, 0]
        feat_hr = feat_dw_hr[:, 1]
        # day of the week
        one_hot_dw = np.zeros((feat_dw.size, 7))
        one_hot_dw[np.arange(feat_dw.size), feat_dw] = 1
        # hour of the day
        one_hot_hr = np.zeros((feat_hr.size, 24))
        one_hot_hr[np.arange(feat_hr.size), feat_hr] = 1
        # concatenate
        feat_one_hot = np.concatenate([one_hot_dw, one_hot_hr], axis=1)

        return feat_one_hot
                                
    def get_index_start_end_alter(self, kh, kw):
        # if alternating, the sum of the index from the beginning and the index from the end is kh*kw - 1
        comp_index = []
        # every column except for the center column
        for cdx in range(kw//2 + 1):
            for rdx in range(kh):
                begin_index = cdx + rdx * kw  # the index from the beginning
                comp_index.append(begin_index)  
                if begin_index == ((kh*kw) // 2):
                    # end at the center hourly block
                    break
                else:
                    end_index = kh * kw - 1 - begin_index  # the index from the end
                    comp_index.append(end_index)  
        return comp_index
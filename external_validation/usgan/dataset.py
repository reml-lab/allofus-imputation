import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader

import sys
sys.path.append("../..")

from utils.data_utils import get_hourly_data, get_multiple_pid, pull_file, FILE_CACHE
from utils.train_utils import feature_padding

from tqdm import tqdm
import pickle
import os
import time

# Note that we only have one split, all the split_idx are set as 0 in the codes
# this case is only for external validation
class AllOfUsDatasetExtValid(Dataset):
    def __init__(self, pid_data, split_idx, dataset="train", ks=(9, 15), pad_full_weeks=True):
        """
        Args:
            - pid_data: (conv_feat, feature_list, step_rate_mean, step_rate_std) for each pid
            - split_idx: split index for the multiple stratified sampling
            - dataset: "train", "valid" or "test"
            - num_split: how many split we use for the stratified sampling
            - ks: (kh, kw) of the context window
            - pad_full_weeks: beyond the one week before and after the current hour, if we would like to use the full week. If not,
                          only the same day of the week from further weeks would be added 
        """
        
        
        self.split_idx = split_idx
        self.dataset = dataset
        self.unfold = nn.Unfold(kernel_size=ks, stride=1)
        self.ks = ks
        self.pad_full_weeks = pad_full_weeks
        self.input_feat_pids, self.step_gt, self.part_mean_sr, self.max_sr,\
            self.sr_mean, self.sr_std, self.pid_ids = self.process(pid_data, ks)
        
    def process(self, pid_data, ks=(9, 15)):

        # if not pull_file(f"{self.dataset}_dataset_split_{self.split_idx}.pkl"):
        # the following process consumes a large memory
        # must apply 8 cpus with 52 GB RAM to finish it
        # so we store them into the google bucket and read
        # it in when it is necessary
        # the memory bottleneck is due to the memory copy of torch.cat
        input_feat_list = []
        step_gt_list = []
        step_rate_mean_list = []
        step_rate_std_list = []
        part_mean_step_rate_list = []
        max_step_rate_list = []
        part_id_list = []
            

        for part_id, (conv_feat, feature_list, step_rate_mean, step_rate_std) in enumerate(tqdm(pid_data)):

            #### get the groudtruth, valid minutes and corresponding masks ####
            # add the batch size 
            conv_feat_tensor = torch.from_numpy(conv_feat[None, ...]).float() # [1, 13, h, w]
            
            #### get the groundtruth ####
            step_gt = conv_feat_tensor[:, [feature_list.index("steps"), 
                                        feature_list.index("valid_minutes"), 
                                        feature_list.index(f"{self.dataset}_mask_split0")], :, :] # [1, 3, h, w]
            step_gt = step_gt.squeeze(0)  # [3, h, w]
            # reshape
            step_gt = step_gt.reshape(step_gt.shape[0], -1).transpose(0, 1)  # [h*w, 3]
            dataset_mask = step_gt[:, -1]  # [h*w]
            step_gt = step_gt[:, :2] # [h*w, 2]

            # we need to use the dataset_mask to mask out all the groundtruth and the context windows which are not 
            # belonged to this dataset (train, valid or test)
            dataset_mask = torch.argwhere(dataset_mask).squeeze(1)  # [h*w]
            # print(f"dataset_mask.shape: {dataset_mask.shape}")
            # print(step_gt.shape)
            step_gt = step_gt[dataset_mask, ...]
            # print(f"step_gt.shape: {step_gt.shape}")
            step_gt_list.append(step_gt)
            
            #### get participant level mean and maximum step rate ####
            # Note that we need to use the training dataset to get the participant mean
            # and the maximum step rate
            #### Different from the dataset used for training, we use the statistics from test here ####
            # since we don't have the training set anymore
            train_step_gt = conv_feat[[feature_list.index("steps"), 
                                    feature_list.index("valid_minutes"), 
                                    feature_list.index(f"test_mask_split0")], :, :] # [3, h, w]
            train_step_gt = torch.from_numpy(train_step_gt)
    
            train_steps = train_step_gt[0,:,:]
            train_valid_min = train_step_gt[1,:,:]
            train_mask = train_step_gt[2,:,:]
            # compute the participant mean on the training dataset of that particular split 
            part_step_rate_mean = (train_steps * train_mask).sum() / (train_valid_min * train_mask).sum()
            # compute the maximum step rate on the training dataset of that particular split
            max_step_rate = (train_steps[train_mask==1] / train_valid_min[train_mask==1]).max()
            
            # repeat them for each context_window
            part_step_rate_mean = part_step_rate_mean.unsqueeze(0).repeat(step_gt.shape[0], 1)
            max_step_rate = max_step_rate.unsqueeze(0).repeat(step_gt.shape[0], 1)

            part_mean_step_rate_list.append(part_step_rate_mean)
            max_step_rate_list.append(max_step_rate)

            #### get the step_rate_mean and step_rate_std ####
            # print(step_rate_mean, step_rate_std, part_step_rate_mean.item(), max_step_rate.item())
            # repeat the step_rate_mean and step_rate_std for each context window
            step_rate_mean = torch.tensor([step_rate_mean]).unsqueeze(0).repeat(step_gt.shape[0], 1)
            step_rate_std = torch.tensor([step_rate_std]).unsqueeze(0).repeat(step_gt.shape[0], 1)
            step_rate_mean_list.append(step_rate_mean)
            step_rate_std_list.append(step_rate_std)

            #### get the part_id ####
            part_id = torch.tensor([part_id]).unsqueeze(0).repeat(step_gt.shape[0], 1)
            part_id_list.append(part_id)

            #### get the input features for the model ####
            # pad the input features in order to save training time
            # Pad the data so that for feature of step rate, there should be valid contexts there to make the time continuous.
            # correctness check has already been contained in feature_padding
            input_feat = conv_feat_tensor[:, :feature_list.index('test_mask_split0'), :, :]
            input_pad_feat = feature_padding(input_feat, kernel_size=ks, device='cpu') # [1, 13, h+2*kh, w+2*kw]
            input_pad_feat = input_pad_feat[:, [feature_list.index("step_rate_norm"),
                                                feature_list.index(f"{self.dataset}_mask_comp_split0"), 
                                                feature_list.index("Day of Week"),
                                                feature_list.index("Hour of Day"),
                                                feature_list.index("time_axis"),
                                                feature_list.index("heart_rate_norm")], :, :]  # [1, 6, h+2*kh, w+2*kw]
            # print(input_pad_feat.shape)
            # print(conv_feat.shape)

            input_pad_feat = self.make_context_windows(input_pad_feat).squeeze(0) # [h*w, 6, kh*kw]
            # select those context windows which are belonged to this dataset
            input_pad_feat = input_pad_feat[dataset_mask, ...]
            # print(f"input_pad_feat.shape: {input_pad_feat.shape}")
            input_feat_list.append(input_pad_feat)

        # concatenate them for all participants in the pid_data
        print("begin concatenating ...")
        part_id_pids = torch.cat(part_id_list, dim=0).int()  # int32
        input_feat_pids = torch.cat(input_feat_list, dim=0).float()  # float32
        step_gt_pids = torch.cat(step_gt_list, dim=0).float()
        part_mean_step_rate_pids = torch.cat(part_mean_step_rate_list, dim=0).float()
        max_step_rate_pids = torch.cat(max_step_rate_list, dim=0).float()
        step_rate_mean_pids = torch.cat(step_rate_mean_list, dim=0).float()
        step_rate_std_pids = torch.cat(step_rate_std_list, dim=0).float()
        print("finish concatenating ...")

        return input_feat_pids, step_gt_pids, part_mean_step_rate_pids, max_step_rate_pids, step_rate_mean_pids, step_rate_std_pids, part_id_pids
            
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

    def __getitem__(self, index):
        # dataloader will add the batch size shape automatically
        return self.input_feat_pids[index, ...], self.step_gt[index, ...], self.part_mean_sr[index, ...], \
               self.max_sr[index, ...], self.sr_mean[index, ...], self.sr_std[index, ...], self.pid_ids[index, ...]#, index_shuffle

    def __len__(self):
        return self.input_feat_pids.shape[0]

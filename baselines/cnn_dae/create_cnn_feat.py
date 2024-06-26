"""
This script is to create the features which are the 2W+1 window around each hourly block
This feature can be used in any 1D based model
"""
import os
import copy
import math
import pickle
import time
import lzma
from tqdm import tqdm
import multiprocessing

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append("../..")

from utils.data_utils import FILE_CACHE, pull_file, get_hourly_data, get_multiple_pid
from utils.train_utils import new_dir

def preprocess_data(df_conv, time_dict, num_split=10):
    """
    Add the computation mask for all splits
    """

    dataset = {i:{} for i in range(num_split)}
    for i in range(num_split):
        dataset[i]["train"] = np.concatenate(time_dict["train"][i])
        dataset[i]["valid"] = np.concatenate(time_dict["valid"][i])
        dataset[i]["test"] = np.concatenate(time_dict["test"][i])
    
    # add train, valid and test mask during the computation (Note that all the above masks are for computing the loss and evaluation metrics but not for computing)
    # Train: test are masked out (i.e. in the context window, there could be training and validation context points, the center chunk needs to be masked out in the model)
    # Valid: test are masked out (i.e. in the context window, there could be training and validation context points, the center chunk needs to be masked out in the model)
    # Test: nothing is masked out (i.e. in the context window, there could be training and validation context points, 
    # and also test points which are the center of other test context windows, the center chunk needs to be masked out in the model)

    # Train: remove all the train and test elements
    # Valid: remove all the valid and test elements
    # Test: remove all the test elements

    for i in range(num_split):
        # train
        df_conv[f"train_mask_comp_split{i}"] = 1
        df_conv.loc[df_conv["time_axis"].isin(dataset[i]["test"]), f"train_mask_comp_split{i}"] = 0
        #df_conv.loc[(df_conv["time_axis"].isin(dataset[i]["train"]))|(df_conv["time_axis"].isin(dataset[i]["test"])), f"train_mask_comp_split{i}"] = 0
        # valid
        df_conv[f"valid_mask_comp_split{i}"] = 1
        #df_conv.loc[(df_conv["time_axis"].isin(dataset[i]["valid"]))|(df_conv["time_axis"].isin(dataset[i]["test"])), f"valid_mask_comp_split{i}"] = 0
        df_conv.loc[df_conv["time_axis"].isin(dataset[i]["test"]), f"valid_mask_comp_split{i}"] = 0
        # test
        df_conv[f"test_mask_comp_split{i}"] = 1
        #df_conv.loc[df_conv["time_axis"].isin(dataset[i]["test"]), f"test_mask_comp_split{i}"] = 0
        # set the mask corresponding to the original missing values as 0
        df_conv.loc[df_conv["step_mask"]==0, [f"train_mask_comp_split{i}", f"valid_mask_comp_split{i}", f"test_mask_comp_split{i}"]] = 0

    # correctness check
    for i in range(num_split):
        assert (np.unique(df_conv.loc[df_conv[f"train_mask_comp_split{i}"]==1, f"train_mask_split{i}"])==np.array([0,1])).all(), f"train_mask_comp_split{i} is wrong!" 
        #assert (np.unique(df_conv.loc[df_conv[f"train_mask_comp_split{i}"]==1, f"train_mask_split{i}"])==np.array([0])).all(), f"train_mask_comp_split{i} is wrong!" 
        assert (np.unique(df_conv.loc[df_conv[f"train_mask_comp_split{i}"]==1, f"valid_mask_split{i}"])==np.array([0,1])).all(), f"train_mask_comp_split{i} is wrong!"
        assert (np.unique(df_conv.loc[df_conv[f"train_mask_comp_split{i}"]==1, f"test_mask_split{i}"])==np.array([0])).all(), f"train_mask_comp_split{i} is wrong!"

        assert (np.unique(df_conv.loc[df_conv[f"valid_mask_comp_split{i}"]==1, f"train_mask_split{i}"])==np.array([0,1])).all(), f"valid_mask_comp_split{i} is wrong!" 
        assert (np.unique(df_conv.loc[df_conv[f"valid_mask_comp_split{i}"]==1, f"valid_mask_split{i}"])==np.array([0,1])).all(), f"valid_mask_comp_split{i} is wrong!"
        #assert (np.unique(df_conv.loc[df_conv[f"valid_mask_comp_split{i}"]==1, f"valid_mask_split{i}"])==np.array([0])).all(), f"valid_mask_comp_split{i} is wrong!"
        assert (np.unique(df_conv.loc[df_conv[f"valid_mask_comp_split{i}"]==1, f"test_mask_split{i}"])==np.array([0])).all(), f"valid_mask_comp_split{i} is wrong!"

        assert (np.unique(df_conv.loc[df_conv[f"test_mask_comp_split{i}"]==1, f"train_mask_split{i}"])==np.array([0,1])).all(), f"test_mask_comp_split{i} is wrong!"
        assert (np.unique(df_conv.loc[df_conv[f"test_mask_comp_split{i}"]==1, f"valid_mask_split{i}"])==np.array([0,1])).all(), f"test_mask_comp_split{i} is wrong!"
        assert (np.unique(df_conv.loc[df_conv[f"test_mask_comp_split{i}"]==1, f"test_mask_split{i}"])==np.array([0,1])).all(), f"test_mask_comp_split{i} is wrong!"
        #assert (np.unique(df_conv.loc[df_conv[f"test_mask_comp_split{i}"]==1, f"test_mask_split{i}"])==np.array([0])).all(), f"test_mask_comp_split{i} is wrong!" 

        assert (np.unique(df_conv.loc[df_conv[f"train_mask_comp_split{i}"]==1, f"step_mask"])==np.array([1])).all(), f"train_mask_comp_split{i} has original missing values!"
        assert (np.unique(df_conv.loc[df_conv[f"valid_mask_comp_split{i}"]==1, f"step_mask"])==np.array([1])).all(), f"valid_mask_comp_split{i} has original missing values!"
        assert (np.unique(df_conv.loc[df_conv[f"test_mask_comp_split{i}"]==1, f"step_mask"])==np.array([1])).all(), f"test_mask_comp_split{i} has original missing values!"

    return df_conv

#### we fill the missing values as zeros for both step rate and heart rate and only fill the 

def pad_fill_values_dayweek_hour(dw_curr, hd_curr, length, direct="backward"):
    """
    Fill the day of the week and hour of the day indicator. We keep missing step rate and heart rate as zeros.
    Args:
        - dw_curr: the current day of the week
        - hd_curr: the current hour of the day
        - length: how many hourly block will go through
        - direct: direction to go, either forward or backward
    """
    # note this time we also need to fill day of the week and hour of the day
    # since they are also the input feature to the cnn model
    #fill_value_list = [] 
    fill_dw_list = []
    fill_hour_list = []
    
    for _ in range(1, length+1):
        if direct == "backward":
            hd_curr = (hd_curr - 1) % 24
            if hd_curr == 23:
                dw_curr = (dw_curr - 1) % 7 
            
        elif direct == "forward":
            hd_curr = (hd_curr + 1) % 24
            if hd_curr == 0:
                dw_curr = (dw_curr + 1) % 7
        
        fill_dw_list.append(dw_curr)
        fill_hour_list.append(hd_curr)
    
    fill_dw_list = np.array(fill_dw_list)
    fill_hour_list = np.array(fill_hour_list)
    
    if direct == "backward":
        return fill_dw_list[::-1], fill_hour_list[::-1]
    else:
        return fill_dw_list, fill_hour_list

def get_feat_all_hourly_blocks(df_exp, split_idx, ctx_len=72):
    """
    Get all the features for the 1D-based algorithm. The features include: normalized step rate, normalized heart rate,
    compute mask, day of the week, hour of the day, steps, valid minutes, true mask
    Args: 
        - df_exp: dataframe of a participant
        - split_idx: split index
        - dayweek_hourly_median: the dictionary recording the median for each dayweek and hour
        - ctx_len: context window length on one side of the current hour (24 means 24 hours before and after the 
                   current hourly block)
    """
    
    df_sr_hd = df_exp.copy(deep=True)

    # we fill all the missing step rate and heart rate as zero
    # and only exterpolate the day of the week and hour of the day
    ext_feat_list = ["step_rate_norm",                     # 0
                     f"train_mask_comp_split{split_idx}",  # 1
                     f"valid_mask_comp_split{split_idx}",  # 2
                     f"test_mask_comp_split{split_idx}",   # 3
                     "Day of Week",                        # 4
                     "Hour of Day",                        # 5
                     "heart_rate_norm"                     # 6
                    ]  # list for extracted feature
    # ext_feat_list += [f"{dataset}_mask_comp_split{split_idx}" for split_idx in range(10) for dataset in ["train", "valid", "test"]]

    out_feat_list = ["steps",
                     f"train_mask_split{split_idx}",
                     f"valid_mask_split{split_idx}",
                     f"test_mask_split{split_idx}", 
                     "valid_minutes"]
    # out_feat_list += [f"{dataset}_mask_split{split_idx}" for split_idx in range(10) for dataset in ["train", "valid", "test"]]

    feat_list = []  # list to store all the features
    gt_list = []
    
    # Here, get the feature in the column format (column after column)
    # instead of the feature in the row format (row after row)

    # for time_axis in tqdm(df_sr_hd["time_axis"].tolist()):
    for hour in tqdm(range(df_sr_hd["Hour of Day"].nunique())):
        for study_day in range(df_sr_hd["Study day"].nunique()):
            
            time_axis = df_sr_hd.loc[(df_sr_hd["Hour of Day"]==hour) & (df_sr_hd["Study day"]==study_day), "time_axis"].item()
            
            # we get the correct context window and get the corresponding features
            if time_axis - ctx_len < df_sr_hd["time_axis"].min():
                df_time = df_sr_hd.loc[(df_sr_hd["time_axis"]>=df_sr_hd["time_axis"].min()) & (df_sr_hd["time_axis"]<=time_axis+ctx_len)]
                feat_hd = df_time[ext_feat_list].to_numpy()
                # pad zeros on the left 
                pad_len = 2*ctx_len+1-feat_hd.shape[0]
                # pad the fill_value on the left
                dw_curr = df_time.iloc[0]["Day of Week"]
                hd_curr = df_time.iloc[0]["Hour of Day"]
                fill_dw_list, fill_hour_list = pad_fill_values_dayweek_hour(dw_curr, hd_curr, pad_len, direct="backward")
                pad_zeros = np.zeros((pad_len, feat_hd.shape[1]))
                pad_zeros[:, 4] = fill_dw_list
                pad_zeros[:, 5] = fill_hour_list
                feat_hd = np.concatenate([pad_zeros, feat_hd], axis=0)

            elif time_axis + ctx_len > df_sr_hd["time_axis"].max():
                df_time = df_sr_hd.loc[(df_sr_hd["time_axis"]>=time_axis-ctx_len) & (df_sr_hd["time_axis"]<=df_sr_hd["time_axis"].max())]
                feat_hd = df_time[ext_feat_list].to_numpy()
                # pad zeros on the right
                pad_len = 2*ctx_len+1-feat_hd.shape[0]
                # pad the fill_value on the right
                dw_curr = df_time.iloc[-1]["Day of Week"]
                hd_curr = df_time.iloc[-1]["Hour of Day"]
                fill_dw_list, fill_hour_list = pad_fill_values_dayweek_hour(dw_curr, hd_curr, pad_len, direct="forward")
                pad_zeros = np.zeros((pad_len, feat_hd.shape[1]))
                pad_zeros[:, 4] = fill_dw_list
                pad_zeros[:, 5] = fill_hour_list
                feat_hd = np.concatenate([feat_hd, pad_zeros], axis=0)

            else:
                df_time = df_sr_hd.loc[(df_sr_hd["time_axis"]>=time_axis-ctx_len) & (df_sr_hd["time_axis"]<=time_axis+ctx_len)]
                feat_hd = df_time[ext_feat_list].to_numpy()
            
            gt_list.append(df_sr_hd.loc[df_sr_hd["time_axis"]==time_axis, out_feat_list].values)
            
            assert feat_hd.shape[0]==2*ctx_len+1, f"pid {df_exp['Participant ID'].unique()[0]} | split {split_idx}| time axis {time_axis} | wrong feat_hd shape!"
           
            feat_list.append(feat_hd[None, ...])  # [1, 2*ctx_len+1, 14]  
    
    hd_feat_mtx = np.concatenate(feat_list, axis=0)
    gt_mtx = np.concatenate(gt_list, axis=0)
    
    assert np.isnan(hd_feat_mtx).any()==False, f"pid {df_exp['Participant ID'].unique()[0]} | split {split_idx} | hd_feat_mtx has nans"   
        
    return hd_feat_mtx.astype("float32"), gt_mtx.astype("float32")


def get_feat_pid_split(pid, ctx_len):
    for split_idx in range(10):
        # read in dataframe of the participant
        df_exp, step_rate_mean, step_rate_std, time_dict = get_hourly_data( pid, num_split=10, ks=(9, 15), start_hour=6, end_hour=22, conv_feat=False, return_time_dict=True)
        # add missing indicator (computational mask) for every split
        df_exp = preprocess_data(df_exp, time_dict) 
        print(f"pid {pid} | split {split_idx} begins ...")
        # create the features for each study day on each split for the KNN input
        hd_feat_mtx_split, gt_mtx_split = get_feat_all_hourly_blocks(df_exp, split_idx, ctx_len)    
        sr_mean_split = np.ones((hd_feat_mtx_split.shape[0], 1)) * step_rate_mean
        sr_std_split = np.ones((hd_feat_mtx_split.shape[0], 1)) * step_rate_std
        print(f"pid {pid} | split {split_idx} gets all hourly blocks")
        # store the results
        pid_raw_feat_dict = {"pid": pid, "split_idx": split_idx, "feat": (hd_feat_mtx_split, gt_mtx_split, sr_mean_split, sr_std_split)}
        filename = f"{FILE_CACHE}/cnn_feat_split{split_idx}/cnn_feat_pid_{pid}_split_{split_idx}.xz"
        print("zipping the data ...")
        with lzma.open(filename, "wb") as f:
            pickle.dump(pid_raw_feat_dict, f)
        print("finish zipping the data")


if __name__ == "__main__":
    ctx_len = 72
    
    pull_file("df_cohort_top100.parquet")  # created by get_cohort_aaai.ipynb
    df_cohort = pd.read_parquet(f"{FILE_CACHE}/df_cohort_top100.parquet")
    pid_list = df_cohort.index.tolist()

    # create folders
    for split_idx in range(10):
        folder_name = f"{FILE_CACHE}/cnn_feat_split{split_idx}"
        new_dir(folder_name)
    
    with multiprocessing.Pool(processes=50) as pool:
        pool.starmap(get_feat_pid_split, list(zip(pid_list, [ctx_len]*len(pid_list))), chunksize=2)
    
    for split_idx in range(10):
        cnn_feat_foldernmae = f"{FILE_CACHE}/cnn_feat_split{split_idx}"
        os.system(f"gsutil -m cp -R {cnn_feat_foldernmae} {os.getenv('WORKSPACE_BUCKET')+'/data/'}")

    


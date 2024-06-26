"""
This script is to create the features which are the 2W+1 window around each hourly block on the external validation set
This feature can be used in any 1D based model
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy

from tqdm import tqdm
import math

import pickle
import time
from scipy import stats

import torch
import torch.nn as nn
import torch.nn.functional as F

import multiprocessing
import lzma

import sys
sys.path.append("../..")

from utils.data_utils import FILE_CACHE, pull_file
from utils.train_utils import new_dir
from extvalid_utils import get_hourly_data
from extvalid_utils import HIGH_MISS_START_END_FILE, HIGH_MISS_SHIFT_FILE, LOW_MISS_START_END_FILE, LOW_MISS_SHIFT_FILE

def preprocess_data(df_conv, time_dict):
    """
    Add the computation mask for all splits
    """
    dataset = {0: {}}
    dataset[0]["test"] = np.array(time_dict["test"][0])
    
    # we only have test here
     # test
    df_conv[f"test_mask_comp_split{0}"] = 1
    #df_conv.loc[df_conv["time_axis"].isin(dataset[i]["test"]), f"test_mask_comp_split{i}"] = 0
    # set the mask corresponding to the original missing values as 0
    # note that hourly blocks out of [start_hour, end_hour] are also visible
    df_conv.loc[df_conv["step_mask"]==0, f"test_mask_comp_split{0}"] = 0

    # correctness check
    assert (len(df_conv.loc[df_conv["step_mask"]==0]) + len(df_conv.loc[df_conv[f"test_mask_comp_split{0}"]==1]))== len(df_conv), "test comp split mask is not correct!"
    
    return df_conv


#### we fill the missing values as zeros for both step rate and heart rate ######
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

def get_feat_all_hourly_blocks(df_exp, ctx_len=72):
    """
    Get all the features for the 1D-based algorithm. The features include: normalized step rate, normalized heart rate,
    compute mask, day of the week, hour of the day, steps, valid minutes, true mask
    Args: 
        - df_exp: dataframe of a participant
        - ctx_len: context window length on one side of the current hour (24 means 24 hours before and after the 
                   current hourly block)
    """
    
    df_sr_hd = df_exp.copy(deep=True)

    # we fill all the missing step rate and heart rate as zero
    # and only exterpolate the day of the week and hour of the day
    ext_feat_list = ["step_rate_norm",                     # 0
                     "test_mask_comp_split0",              # 1
                     "Day of Week",                        # 2
                     "Hour of Day",                        # 3
                     "heart_rate_norm"                     # 4
                    ]  # list for extracted feature
    
    out_feat_list = ["steps",
                     "test_mask_split0", 
                     "valid_minutes"]
    
    feat_list = []  # list to store all the features
    gt_list = []
    
    # Here, get the feature in the column format (column after column)
    # instead of the feature in the row format (row after row)

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
                pad_zeros[:, 2] = fill_dw_list
                pad_zeros[:, 3] = fill_hour_list
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
                pad_zeros[:, 2] = fill_dw_list
                pad_zeros[:, 3] = fill_hour_list
                feat_hd = np.concatenate([feat_hd, pad_zeros], axis=0)

            else:
                df_time = df_sr_hd.loc[(df_sr_hd["time_axis"]>=time_axis-ctx_len) & (df_sr_hd["time_axis"]<=time_axis+ctx_len)]
                feat_hd = df_time[ext_feat_list].to_numpy()
            
            gt_list.append(df_sr_hd.loc[df_sr_hd["time_axis"]==time_axis, out_feat_list].values)
            
            assert feat_hd.shape[0]==2*ctx_len+1, f"pid {df_exp['Participant ID'].unique()[0]} | time axis {time_axis} | wrong feat_hd shape!"
           
            feat_list.append(feat_hd[None, ...])  # [1, 2*ctx_len+1, 14]  
    
    hd_feat_mtx = np.concatenate(feat_list, axis=0)
    gt_mtx = np.concatenate(gt_list, axis=0)
    
    assert np.isnan(hd_feat_mtx).any()==False, f"pid {df_exp['Participant ID'].unique()[0]} | hd_feat_mtx has nans"   
        
    return hd_feat_mtx.astype("float32"), gt_mtx.astype("float32")


def get_feat_pid(pid, ctx_len, df_start_end_day, df_best_shift_dw, if_high_miss):
    # read in dataframe of the participant
    df_exp, step_rate_mean, step_rate_std, time_dict = get_hourly_data( pid, df_start_end_day, df_best_shift_dw, start_hour=6, end_hour=22, conv_feat=False, return_time_dict=True)
    
    # add missing indicator (computational mask) for every split
    df_exp = preprocess_data(df_exp, time_dict) 
    print(f"pid {pid} begins ...")
    # create the features for each study day on each split for the KNN input
    hd_feat_mtx, gt_mtx = get_feat_all_hourly_blocks(df_exp, ctx_len)    
    sr_mean = np.ones((hd_feat_mtx.shape[0], 1)) * step_rate_mean
    sr_std = np.ones((hd_feat_mtx.shape[0], 1)) * step_rate_std
    print(f"pid {pid} gets all hourly blocks")
    # store the results
    pid_raw_feat_dict = {"pid": pid, "feat": (hd_feat_mtx, gt_mtx, sr_mean, sr_std)}
    if if_high_miss:
        filename = f"{FILE_CACHE}/cnn_feat_extvalid_high_miss/cnn_feat_pid_{pid}.xz"
    else:
        filename = f"{FILE_CACHE}/cnn_feat_extvalid_low_miss/cnn_feat_pid_{pid}.xz"
    print("zipping the data ...")
    with lzma.open(filename, "wb") as f:
        pickle.dump(pid_raw_feat_dict, f)
    print("finish zipping the data")


if __name__ == "__main__":
    ctx_len = 72

    if_high_miss = True

    if if_high_miss:
        # 100 participants
        pull_file(HIGH_MISS_START_END_FILE) # get the start day and end day file
        pull_file(HIGH_MISS_SHIFT_FILE) # get the shift of day of the week file
        df_start_end_day = pd.read_parquet(f"{FILE_CACHE}/{HIGH_MISS_START_END_FILE}")
        df_best_shift_dw = pd.read_parquet(f"{FILE_CACHE}/{HIGH_MISS_SHIFT_FILE}") 
        # build folder to store features
        new_dir(f"{FILE_CACHE}/cnn_feat_extvalid_high_miss")
        print("we are doing high miss rate!")
    else:
        # 400 participants
        pull_file(LOW_MISS_START_END_FILE) # get the start day and end day file
        pull_file(LOW_MISS_SHIFT_FILE) # get the shift of day of the week file
        df_start_end_day = pd.read_parquet(f"{FILE_CACHE}/{LOW_MISS_START_END_FILE}")
        df_best_shift_dw = pd.read_parquet(f"{FILE_CACHE}/{LOW_MISS_SHIFT_FILE}") 
        # build folder to store the features
        new_dir(f"{FILE_CACHE}/cnn_feat_extvalid_low_miss")
        print("we are doing low miss rate!")

    pid_list = df_start_end_day.index.tolist()

    df_start_end_day_list = [df_start_end_day] * len(pid_list) 
    df_best_shift_dw_list = [df_best_shift_dw] * len(pid_list) 
    if_high_miss_list = [if_high_miss] * len(pid_list)
    
    with multiprocessing.Pool(processes=30) as pool:
        pool.starmap(get_feat_pid, list(zip(pid_list, [ctx_len]*len(pid_list), df_start_end_day_list, df_best_shift_dw_list, if_high_miss_list)))
    
    


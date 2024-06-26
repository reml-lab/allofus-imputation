"""
Create Local Activity Profile Representation features
"""
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

import argparse
import pickle
import time
from tqdm import tqdm
import copy
import os
import multiprocessing
import lzma

from utils.data_utils import pull_file, FILE_CACHE, get_hourly_data
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


# get the median normalized step rate for each day of week and hour of day (7 * 24 factors)
def get_dayweek_hour_median(df_part, split_idx):
    """
    Fill with day of the week and hour of the day median (median of the all valid hourly blocks)
    The median are obtained from the train and valid, not from the test
    Args:
        - df_part: the dataframe of the participant
        - split_idx: split index
    """
    df_exp = df_part.copy(deep=True)
    # Note that we need to set all the step_mask before 6:00 and after 23:00 as 1 here
    # since for padding the step rate values, we need to use these step rates
    df_exp.loc[((df_exp["Hour of Day"]<6)|(df_exp["Hour of Day"]>22)) & (df_exp["valid_minutes"]>0), "step_mask"] = 1

    # split the dataframe into train, valid and test
    # Note that train, valid and test accounts for all the valid hourly blocks between 6:00 ~ 22:00
    # Note here df_train_valid also includes these blocks before 6:00 and after 22:00
    df_train_valid = df_exp.loc[(df_exp[f"test_mask_split{split_idx}"]==0)]

    dayweek_hourly_median = {}
    for day in range(7):
        dayweek_hourly_median[day] = {}
        df_dayweek = df_train_valid.loc[df_train_valid["Day of Week"]==day]
        for hour in range(24):
            df_dayweek_hour = df_dayweek.loc[df_dayweek["Hour of Day"]==hour]
            # if there is no such day of week and hour of day, then record the median as the participant level median
            # also we compute the median on the level of normalized step rate instead of step rate
            if len(df_dayweek_hour) == 0:
                dayweek_hourly_median[day][hour] = df_train_valid.loc[df_train_valid["step_mask"]==1, "step_rate_norm"].median()
            else:
                dayweek_hourly_median[day][hour] = df_dayweek_hour.loc[df_dayweek_hour["step_mask"]==1, "step_rate_norm"].median()

    return dayweek_hourly_median


def pad_fill_values_dayweek_hour(dw_curr, hd_curr, length, dw_hd_med_dict, direct="backward"):
    """
    Pad the fill values based on the current dayweek and hour. The fill value is dw_hd_med_dict, and we go from the current
    day of the week and hour of the day backward or foward (depends on which side to pad) by length hourly blocks
    Args:
        - dw_curr: the current day of the week
        - hd_curr: the current hour of the day
        - length: how many hourly block will go through
        - dw_hd_med_dict: the dictionary storing the median value of normalized step rate for each dayweek and hour
        - direct: direction to go, either forward or backward
    """
    fill_value_list = [] 
    
    for _ in range(1, length+1):
        if direct == "backward":
            hd_curr = (hd_curr - 1) % 24
            if hd_curr == 23:
                dw_curr = (dw_curr - 1) % 7 
            
        elif direct == "forward":
            hd_curr = (hd_curr + 1) % 24
            if hd_curr == 0:
                dw_curr = (dw_curr + 1) % 7
        
        fill_value_list.append(dw_hd_med_dict[dw_curr][hd_curr])
    
    fill_value_list = np.array(fill_value_list)
    
    if direct == "backward":
        return fill_value_list[::-1]
    else:
        return fill_value_list
    

def get_feat_all_hourly_blocks(df_sr_hd, split_idx, dayweek_hourly_median, ctx_len=72, start_time=6, end_time=22):
    """
    Get all the necessary features for the KNN algorithm. The features include: normalized step rate, compute
    mask, true mask, time axis, steps, valid minutes
    Args: 
        - df_sr_hd: dataframe of a participant
        - split_idx: split index
        - dayweek_hourly_median: the dictionary recording the median for each dayweek and hour
        - ctx_len: context window length on one side of the current hour (24 means 24 hours before and after the 
                   current hourly block)
    """
    # fill in the potential filling values for every hourly block
    df_sr_hd["fill_value"] = df_sr_hd.apply(lambda x: dayweek_hourly_median[x["Day of Week"]][x["Hour of Day"]], axis=1)
    # note that the median is computed on the level of normalized step rate
    # reason why we don't use the unnormalized step rate is that we fill zero for both step rate and normalized step rate
    # when it is missing, then if we normalizing it again, it would be wrong
    ext_feat_list = ["step_rate_norm",                     # 0
                     f"train_mask_comp_split{split_idx}",  # 1
                     f"valid_mask_comp_split{split_idx}",  # 2
                     f"test_mask_comp_split{split_idx}",   # 3
                    ]  # list for extracted feature

    feat_list_train = []  # list to store all the features in the train dataset
    feat_list_valid = []  # list to store all the features in the valid dataset
    feat_list_test = []  # list to store all the features in the test dataset
    # fill_value_list = []  # list to store all the possible filling values (if some hourly block is missing, then we fill it with the median of that dw + hd)
    # fill_value_center_block_list = []  # list to store all the possible filling values for each possible center hourly block
    # Here, get the feature in the column format (column after column)
    # instead of the feature in the row format (row after row)
    
    ctx_len = ctx_len + 4  # counts from the center of that day
    
    # for time_axis in tqdm(df_sr_hd["time_axis"].tolist()):
    for hour in range(start_time, end_time+1):
        for study_day in range(df_sr_hd["Study day"].nunique()):
            time_axis_ctr = df_sr_hd.loc[(df_sr_hd["Hour of Day"]==hour) & (df_sr_hd["Study day"]==study_day), "time_axis"].item()
            df_ctw = df_sr_hd.copy()
            dw_ctr = df_ctw.loc[df_ctw["time_axis"]==time_axis_ctr, "Day of Week"].item()
            hour_ctr = hour
            #### important ####
            # we need to assign the center hourly block to the median, otherwise, there would be the groundtruth leakage
            df_ctw.loc[df_ctw["time_axis"]==time_axis_ctr, "step_rate_norm"] = df_ctw.loc[df_ctw["time_axis"]==time_axis_ctr, "fill_value"]
            # we get all the necessary day difference for this context window
            day_gap_index_list = np.concatenate([np.arange(-5,0) * 7, np.arange(-6, 8), np.arange(2, 6) * 7])
            nsr_ctw_list = []  # store all the nsr feature for this context window
            
            for day_gap in day_gap_index_list:
                time_axis = time_axis_ctr + day_gap * 24 
                
                # we get the correct context window and get the corresponding features
                if time_axis - ctx_len < df_ctw["time_axis"].min():
                    df_time = df_ctw.loc[(df_ctw["time_axis"]>=df_ctw["time_axis"].min()) & (df_ctw["time_axis"]<=time_axis+ctx_len)]
                    feat_hd = df_time[ext_feat_list].to_numpy()
                    # pad zeros on the left 
                    pad_len = 2*ctx_len+1-feat_hd.shape[0]
                    feat_hd = np.concatenate([np.zeros((pad_len, feat_hd.shape[1])), feat_hd], axis=0)
                    # pad the fill_value on the left
                    if time_axis + ctx_len < df_ctw["time_axis"].min():
                        # we set the hack time axis as the next of the last one in that feature window
                        # and do the backward fill
                        hour_hack = (hour_ctr + day_gap * 24 + ctx_len +1) % 24
                        dw_hack = (dw_ctr + (day_gap * 24 + ctx_len + 1) // 24) % 7
                        # if last element is still be smaller than the minimum time axis, 
                        # then we move the current pointer to the next one of
                        # the last one in the context window and backward fill
                        fill_value = pad_fill_values_dayweek_hour(dw_hack, hour_hack, 2*ctx_len+1, dayweek_hourly_median, direct="backward")
                    else:
                        fill_value = df_time["fill_value"].to_numpy()
                        dw_curr = df_time.iloc[0]["Day of Week"]
                        hd_curr = df_time.iloc[0]["Hour of Day"]
                        fill_value_pad = pad_fill_values_dayweek_hour(dw_curr, hd_curr, pad_len, dayweek_hourly_median, direct="backward")
                        fill_value = np.concatenate([fill_value_pad, fill_value])
                
                elif time_axis + ctx_len > df_ctw["time_axis"].max():
                    df_time = df_ctw.loc[(df_ctw["time_axis"]>=time_axis-ctx_len) & (df_ctw["time_axis"]<=df_ctw["time_axis"].max())]
                    feat_hd = df_time[ext_feat_list].to_numpy()
                    # pad zeros on the right
                    pad_len = 2*ctx_len+1-feat_hd.shape[0]
                    feat_hd = np.concatenate([feat_hd, np.zeros((pad_len, feat_hd.shape[1]))], axis=0)
                    # pad the fill_value on the right
                    if time_axis - ctx_len > df_ctw["time_axis"].max():
                        # we set the hack time axis as the next of the last one in that feature window
                        # and do the backward fill
                        hour_hack = (hour_ctr + day_gap * 24 + ctx_len +1) % 24
                        dw_hack = (dw_ctr + (day_gap * 24 + ctx_len + 1) // 24) % 7
                        # if first element is still be larger than the minimum time axis, 
                        # then we move the current pointer to the next one of
                        # the last one in the context window and backward fill
                        fill_value = pad_fill_values_dayweek_hour(dw_hack, hour_hack, 2*ctx_len+1, dayweek_hourly_median, direct="backward")
                    else:
                        fill_value = df_time["fill_value"].to_numpy()
                        dw_curr = df_time.iloc[-1]["Day of Week"]
                        hd_curr = df_time.iloc[-1]["Hour of Day"]
                        fill_value_pad = pad_fill_values_dayweek_hour(dw_curr, hd_curr, pad_len, dayweek_hourly_median, direct="forward")
                        fill_value = np.concatenate([fill_value, fill_value_pad])
                
                else:
                    df_time = df_ctw.loc[(df_ctw["time_axis"]>=time_axis-ctx_len) & (df_ctw["time_axis"]<=time_axis+ctx_len)]
                    feat_hd = df_time[ext_feat_list].to_numpy()
                    fill_value = df_time["fill_value"].to_numpy()
                    
                assert feat_hd.shape[0]==2*ctx_len+1, f"split {split_idx} | time axis {time_axis} | wrong feat_hd shape!"
                assert fill_value.shape[0]==2*ctx_len+1, f"split {split_idx} | time axis {time_axis} | wrong fill_value shape"
            
                feat_hd_train = feat_hd[:, 0] * feat_hd[:, 1] + fill_value * (1-feat_hd[:, 1])
                feat_hd_valid = feat_hd[:, 0] * feat_hd[:, 2] + fill_value * (1-feat_hd[:, 2])
                feat_hd_test = feat_hd[:, 0] * feat_hd[:, 3] + fill_value * (1 - feat_hd[:, 3])
                feat_hd = np.concatenate([feat_hd_train[...,None], feat_hd_valid[...,None], feat_hd_test[...,None]], axis=-1)
                # here, the shape of feat_hd is [153, 3]
                nsr_ctw_list.append(feat_hd[None, ...])  # [1, 153, 3]
            # concatenate for this context window horizontally based on study day in the context window
            feat_hd_ctw = np.concatenate(nsr_ctw_list, axis=0)  # [23, 153, 3]
            
            if df_sr_hd.loc[(df_sr_hd["Hour of Day"]==hour) & (df_sr_hd["Study day"]==study_day), f"train_mask_split{split_idx}"].item() == 1:
                feat_list_train.append(feat_hd_ctw[:, :, 0][None,...]) # [1, 23, 153]
            elif df_sr_hd.loc[(df_sr_hd["Hour of Day"]==hour) & (df_sr_hd["Study day"]==study_day), f"valid_mask_split{split_idx}"].item() == 1:
                feat_list_valid.append(feat_hd_ctw[:,:,1][None, ...])
            elif df_sr_hd.loc[(df_sr_hd["Hour of Day"]==hour) & (df_sr_hd["Study day"]==study_day), f"test_mask_split{split_idx}"].item() == 1:
                feat_list_test.append(feat_hd_ctw[:,:,2][None, ...])  # [1, 23, 153]  
            
    hd_feat_mtx_train = np.concatenate(feat_list_train, axis=0)
    hd_feat_mtx_valid = np.concatenate(feat_list_valid, axis=0)
    hd_feat_mtx_test = np.concatenate(feat_list_test, axis=0)
    
    assert np.isnan(hd_feat_mtx_train).any()==False, f"split {split_idx} | hd_feat_mtx_train has nans"  
    assert np.isnan(hd_feat_mtx_valid).any()==False, f"split {split_idx} | hd_feat_mtx_valid has nans" 
    assert np.isnan(hd_feat_mtx_test).any()==False, f"split {split_idx} | hd_feat_mtx_test has nans"  
        
    return hd_feat_mtx_train.astype("float32"), hd_feat_mtx_valid.astype("float32"), hd_feat_mtx_test.astype("float32")


def get_feat_pid_split(pid, ctx_len):
    # read in dataframe of the participant
    df_exp, _, _, time_dict = get_hourly_data( pid, num_split=10, ks=(9, 15), start_hour=6, end_hour=22, conv_feat=False, return_time_dict=True)
    # add missing indicator (computational mask) for every split
    df_exp = preprocess_data(df_exp, time_dict) 
    for split_idx in range(10):
        start_time = time.time()
        print(f"pid {pid} | split {split_idx} begins ...")
        # get the dictionary of fill values
        dw_hd_med_dict = get_dayweek_hour_median(df_exp, split_idx)
        # get the raw step rate feature for the deep model
        hd_feat_mtx_train, hd_feat_mtx_valid, hd_feat_mtx_test = get_feat_all_hourly_blocks(df_exp.copy(), 
                                                                                            split_idx, 
                                                                                            dw_hd_med_dict, 
                                                                                            ctx_len=ctx_len)
        # write the features into the file
        PATH = f"{FILE_CACHE}/lapr_feat_split{split_idx}"
        with lzma.open(f"{PATH}/pid_{pid}_split_{split_idx}_train.xz", "wb") as f:
            pickle.dump(hd_feat_mtx_train, f)
        with lzma.open(f"{PATH}/pid_{pid}_split_{split_idx}_valid.xz", "wb") as f:
            pickle.dump(hd_feat_mtx_valid, f)
        with lzma.open(f"{PATH}/pid_{pid}_split_{split_idx}_test.xz", "wb") as f:
            pickle.dump(hd_feat_mtx_test, f)

        print(f"pid {pid} | split {split_idx} | finishes in {(time.time() - start_time):.2f} seconds")

        
if __name__ == "__main__":

    # build folder to store the loss history and the best model
    for split_idx in range(10):
        new_dir(f"{FILE_CACHE}/lapr_feat_split{split_idx}")

    ctx_len = 72
    
    pull_file("df_cohort_top100.parquet")  # created by get_cohort_aaai.ipynb
    df_cohort = pd.read_parquet(f"{FILE_CACHE}/df_cohort_top100.parquet")
    pid_list = df_cohort.index.tolist()
    
    with multiprocessing.Pool() as pool:
        pool.starmap(get_feat_pid_split, list(zip(pid_list, [ctx_len]*len(pid_list))), chunksize=2)
    
"""
Simple filling methods
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import os
import time
import copy
import re
import math
import pickle
from tqdm import tqdm

import sys
sys.path.append("../..")
from utils import data_utils, train_utils

import warnings
warnings.filterwarnings("ignore")


def participant_average_mean(df_exp, num_splits=10, verbose=False):
    """
    Fill with the participant level mean step rate (sum of steps / sum of valid minutes). Since we only
    care about 6:00 ~ 22:00, for participant level mean, we only compute the mean from the hourly blocks 
    which is between 6:00 ~ 22:00.
    Args:
        - df_exp: the dataframe of the participant
        - num_splits: number of splits in the data
        - verbose: wheter to print the metrics we computed for each split of this participant
    """
    # construct the dictionary to record and test performance
    test_loss = {"mae": [], "mse": [], "rmse":[], "num_hourly_blocks": []}
    
    if verbose:
        print(f"participant id = {df_exp['Participant ID'].unique().item()}")
        print("-" * 100)
    
    for i in range(num_splits):

        # Note that train, valid and test accounts for all the valid hourly blocks between 6:00 ~ 22:00
        df_train_valid = df_exp.loc[(df_exp[f"train_mask_split{i}"]==1)|(df_exp[f"valid_mask_split{i}"]==1)]
        df_test = df_exp.loc[df_exp[f"test_mask_split{i}"]==1]

        # train
        part_step_rate_mean = df_train_valid["steps"].sum() / df_train_valid["valid_minutes"].sum()

        # test
        df_test["step_rate_pred"] = part_step_rate_mean
        df_test["steps_pred"] = df_test["step_rate_pred"] * df_test["valid_minutes"]
        test_mae = np.mean(np.abs(df_test["steps"].values - df_test["steps_pred"].values))
        test_mse = np.mean((df_test["steps"].values - df_test["steps_pred"].values)**2)
        
        # record the performance
        test_loss["mae"].append(test_mae)
        test_loss["mse"].append(test_mse)
        test_loss["rmse"].append(np.sqrt(test_mse))
        
        # also record the number of test hourly blocks
        test_loss["num_hourly_blocks"].append(len(df_test))
        
        if verbose:
            print(f"split {i} | test mae: {test_mae:.2f} | test mse: {test_mse:.2f} | test rmse: {np.sqrt(test_mse):.2f}")

    # record the pid
    test_loss["participant_id"] = df_exp["Participant ID"].unique().item()
            
    if verbose:
        print()
        print(f'mean(std) | test mae: {np.mean(test_loss["mae"]):.2f}({np.std(test_loss["mae"]):.2f})')
        print(f'mean(std) | test mse: {np.mean(test_loss["mse"]):.2f}({np.std(test_loss["mse"]):.2f})')
        print(f'mean(std) | test rmse: {np.mean(test_loss["rmse"]):.2f}({np.std(test_loss["rmse"]):.2f})')

    return test_loss
    
def dayweek_average_mean(df_exp, num_splits=10, verbose=False):
    """
    Fill with day of the week mean (sum of steps / sum of valid minutes)
    Args:
        - df_exp: the dataframe of the participant
        - num_splits: number of splits in the data
        - verbose: wheter to print the metrics we computed for each split of this participant
    """
    
    # construct the dictionary to record and test performance
    test_loss = {"mae": [], "mse": [], "rmse":[], "num_hourly_blocks": []}
    
    if verbose:
        print(f"participant id = {df_exp['Participant ID'].unique().item()}")
        print("-" * 100)
    
    for i in range(num_splits):
        # split the dataframe into train, valid and test
        # Note that train, valid and test accounts for all the valid hourly blocks between 6:00 ~ 22:00
        df_train_valid = df_exp.loc[(df_exp[f"train_mask_split{i}"]==1)|(df_exp[f"valid_mask_split{i}"]==1)]
        df_test = df_exp.loc[df_exp[f"test_mask_split{i}"]==1]

        # train
        dayweek_level_mean = {}
        for day in range(7):    
            df_dayweek = df_train_valid.loc[df_train_valid["Day of Week"]==day]
            # if some day of week is missing, we use the participant level mean
            if len(df_dayweek) == 0:
                dayweek_level_mean[day] = df_train_valid["steps"].sum() / df_train_valid["valid_minutes"].sum()
            else:
                dayweek_level_mean[day] = df_dayweek["steps"].sum() / df_dayweek["valid_minutes"].sum()

        # test
        df_test["step_rate_pred"] = df_test.apply(lambda x: dayweek_level_mean[x["Day of Week"]], axis=1)
        df_test["steps_pred"] = df_test["step_rate_pred"] * df_test["valid_minutes"]
        test_mae = np.mean(np.abs(df_test["steps"].values - df_test["steps_pred"].values))
        test_mse = np.mean((df_test["steps"].values - df_test["steps_pred"].values)**2)

        # record the performance on the test set
        test_loss["mae"].append(test_mae)
        test_loss["mse"].append(test_mse)
        test_loss["rmse"].append(np.sqrt(test_mse))

        # also record the number of test hourly blocks
        test_loss["num_hourly_blocks"].append(len(df_test))
        
        if verbose:
            print(f"split {i} | test mae: {test_mae:.2f} | test mse: {test_mse:.2f} | test rmse: {np.sqrt(test_mse):.2f}")

    # record the pid
    test_loss["participant_id"] = df_exp["Participant ID"].unique().item()
            
    if verbose:
        print()
        print(f'mean(std) | test mae: {np.mean(test_loss["mae"]):.2f}({np.std(test_loss["mae"]):.2f})')
        print(f'mean(std) | test mse: {np.mean(test_loss["mse"]):.2f}({np.std(test_loss["mse"]):.2f})')
        print(f'mean(std) | test rmse: {np.mean(test_loss["rmse"]):.2f}({np.std(test_loss["rmse"]):.2f})')

    return test_loss

def hour_average_mean(df_exp, num_splits=10, verbose=False):
    """
    Fill with hour of the day mean (sum of steps / sum of valid minutes)
    Args:
        - df_exp: the dataframe of the participant
        - num_splits: number of splits in the data
        - verbose: wheter to print the metrics we computed for each split of this participant
    """
    
    # construct the dictionary to record and test performance
    test_loss = {"mae": [], "mse": [], "rmse":[], "num_hourly_blocks": []}
    
    if verbose:
        print(f"participant id = {df_exp['Participant ID'].unique().item()}")
        print("-" * 100)
    
    for i in range(num_splits):
        # split the dataframe into train, valid and test
        # Note that train, valid and test accounts for all the valid hourly blocks between 6:00 ~ 22:00
        df_train_valid = df_exp.loc[(df_exp[f"train_mask_split{i}"]==1)|(df_exp[f"valid_mask_split{i}"]==1)]
        df_test = df_exp.loc[df_exp[f"test_mask_split{i}"]==1]

        # train
        hour_level_mean = {}
        for hour in range(24):
            df_hour = df_train_valid.loc[df_train_valid["Hour of Day"]==hour]
            if len(df_hour) == 0:
                hour_level_mean[hour] = df_train_valid["steps"].sum() / df_train_valid["valid_minutes"].sum()
            else:
                hour_level_mean[hour] = df_hour["steps"].sum() / df_hour["valid_minutes"].sum()
                
        # test
        df_test["step_rate_pred"] = df_test.apply(lambda x: hour_level_mean[x["Hour of Day"]], axis=1)
        df_test["steps_pred"] = df_test["step_rate_pred"] * df_test["valid_minutes"]
        test_mae = np.mean(np.abs(df_test["steps"].values - df_test["steps_pred"].values))
        test_mse = np.mean((df_test["steps"].values - df_test["steps_pred"].values)**2)

        # record the performance on the test set
        test_loss["mae"].append(test_mae)
        test_loss["mse"].append(test_mse)
        test_loss["rmse"].append(np.sqrt(test_mse))

        # also record the number of test hourly blocks
        test_loss["num_hourly_blocks"].append(len(df_test))
        
        if verbose:
            print(f"split {i} | test mae: {test_mae:.2f} | test mse: {test_mse:.2f} | test rmse: {np.sqrt(test_mse):.2f}")

    # record the pid
    test_loss["participant_id"] = df_exp["Participant ID"].unique().item()
            
    if verbose:
        print()
        print(f'mean(std) | test mae: {np.mean(test_loss["mae"]):.2f}({np.std(test_loss["mae"]):.2f})')
        print(f'mean(std) | test mse: {np.mean(test_loss["mse"]):.2f}({np.std(test_loss["mse"]):.2f})')
        print(f'mean(std) | test rmse: {np.mean(test_loss["rmse"]):.2f}({np.std(test_loss["rmse"]):.2f})')

    return test_loss

def dayweek_hour_average_mean(df_exp, num_splits=10, verbose=False):
    """
    Fill with day of the week and hour of the day mean (sum of steps / sum of valid minutes)
    Args:
        - df_exp: the dataframe of the participant
        - num_splits: number of splits in the data
        - verbose: wheter to print the metrics we computed for each split of this participant
    """
    
    # construct the dictionary to record and test performance
    test_loss = {"mae": [], "mse": [], "rmse":[], "num_hourly_blocks": []}
    
    if verbose:
        print(f"participant id = {df_exp['Participant ID'].unique().item()}")
        print("-" * 100)
    
    for i in range(num_splits):
        # split the dataframe into train, valid and test
        # Note that train, valid and test accounts for all the valid hourly blocks between 6:00 ~ 22:00
        df_train_valid = df_exp.loc[(df_exp[f"train_mask_split{i}"]==1)|(df_exp[f"valid_mask_split{i}"]==1)]
        df_test = df_exp.loc[df_exp[f"test_mask_split{i}"]==1]

        # train
        dayweek_hourly_mean = {}
        for day in range(7):
            dayweek_hourly_mean[day] = {}
            df_dayweek = df_train_valid.loc[df_train_valid["Day of Week"]==day]
            for hour in range(24):
                df_dayweek_hour = df_dayweek.loc[df_dayweek["Hour of Day"]==hour]
                if len(df_dayweek_hour) == 0:
                    dayweek_hourly_mean[day][hour] = df_train_valid["steps"].sum() / df_train_valid["valid_minutes"].sum()
                else:
                    dayweek_hourly_mean[day][hour] = df_dayweek_hour["steps"].sum() / df_dayweek_hour["valid_minutes"].sum()

        # test
        df_test["step_rate_pred"] = df_test.apply(lambda x: dayweek_hourly_mean[x["Day of Week"]][x["Hour of Day"]], axis=1)
        df_test["steps_pred"] = df_test["step_rate_pred"] * df_test["valid_minutes"]
        test_mae = np.mean(np.abs(df_test["steps"].values - df_test["steps_pred"].values))
        test_mse = np.mean((df_test["steps"].values - df_test["steps_pred"].values)**2)
        
        # record the performance on the test set
        test_loss["mae"].append(test_mae)
        test_loss["mse"].append(test_mse)
        test_loss["rmse"].append(np.sqrt(test_mse))

        # also record the number of test hourly blocks
        test_loss["num_hourly_blocks"].append(len(df_test))
        
        if verbose:
            print(f"split {i} | test mae: {test_mae:.2f} | test mse: {test_mse:.2f} | test rmse: {np.sqrt(test_mse):.2f}")

    # record the pid
    test_loss["participant_id"] = df_exp["Participant ID"].unique().item()
            
    if verbose:
        print()
        print(f'mean(std) | test mae: {np.mean(test_loss["mae"]):.2f}({np.std(test_loss["mae"]):.2f})')
        print(f'mean(std) | test mse: {np.mean(test_loss["mse"]):.2f}({np.std(test_loss["mse"]):.2f})')
        print(f'mean(std) | test rmse: {np.mean(test_loss["rmse"]):.2f}({np.std(test_loss["rmse"]):.2f})')

    return test_loss

def participant_mean(df_exp, num_splits=10, verbose=False):
    """
    Fill with the mean of all of the valid hourly blocks in the participant data. Since we only care about 
    6:00 ~ 22:00, we only include 6:00 ~ 22:00 data to compute the mean
    Args:
        - df_exp: the dataframe of the participant
        - num_splits: number of splits in the data
        - verbose: wheter to print the metrics we computed for each split of this participant
    """
    
    # construct the dictionary to record and test performance
    test_loss = {"mae": [], "mse": [], "rmse":[], "num_hourly_blocks": []}
    
    if verbose:
        print(f"participant id = {df_exp['Participant ID'].unique().item()}")
        print("-" * 100)
    
    for i in range(num_splits):
        # split the dataframe into train, valid and test
        # Note that train, valid and test accounts for all the valid hourly blocks between 6:00 ~ 22:00
        df_train_valid = df_exp.loc[(df_exp[f"train_mask_split{i}"]==1)|(df_exp[f"valid_mask_split{i}"]==1)]
        df_test = df_exp.loc[df_exp[f"test_mask_split{i}"]==1]

        # train
        part_step_rate_mean = df_train_valid.loc[df_train_valid["step_mask"]==1, "step_rate"].mean()

        # test
        df_test["step_rate_pred"] = part_step_rate_mean
        df_test["steps_pred"] = df_test["step_rate_pred"] * df_test["valid_minutes"]
        test_mae = np.mean(np.abs(df_test["steps"].values - df_test["steps_pred"].values))
        test_mse = np.mean((df_test["steps"].values - df_test["steps_pred"].values)**2)
        
        # record the performance on the test set
        test_loss["mae"].append(test_mae)
        test_loss["mse"].append(test_mse)
        test_loss["rmse"].append(np.sqrt(test_mse))

        # also record the number of test hourly blocks
        test_loss["num_hourly_blocks"].append(len(df_test))
        
        if verbose:
            print(f"split {i} | test mae: {test_mae:.2f} | test mse: {test_mse:.2f} | test rmse: {np.sqrt(test_mse):.2f}")

    # record the pid
    test_loss["participant_id"] = df_exp["Participant ID"].unique().item()
            
    if verbose:
        print()
        print(f'mean(std) | test mae: {np.mean(test_loss["mae"]):.2f}({np.std(test_loss["mae"]):.2f})')
        print(f'mean(std) | test mse: {np.mean(test_loss["mse"]):.2f}({np.std(test_loss["mse"]):.2f})')
        print(f'mean(std) | test rmse: {np.mean(test_loss["rmse"]):.2f}({np.std(test_loss["rmse"]):.2f})')

    return test_loss

def dayweek_mean(df_exp, num_splits=10, verbose=False):
    """
    Fill with day of the week mean (mean of the all valid hourly blocks)
    Args:
        - df_exp: the dataframe of the participant
        - num_splits: number of splits in the data
        - verbose: wheter to print the metrics we computed for each split of this participant
    """
    
    # construct the dictionary to record and test performance
    test_loss = {"mae": [], "mse": [], "rmse":[], "num_hourly_blocks": []}
    
    if verbose:
        print(f"participant id = {df_exp['Participant ID'].unique().item()}")
        print("-" * 100)
    
    for i in range(num_splits):
        # split the dataframe into train, valid and test
        # Note that train, valid and test accounts for all the valid hourly blocks between 6:00 ~ 22:00
        df_train_valid = df_exp.loc[(df_exp[f"train_mask_split{i}"]==1)|(df_exp[f"valid_mask_split{i}"]==1)]
        df_test = df_exp.loc[df_exp[f"test_mask_split{i}"]==1]

        # train
        dayweek_level_mean = {}
        for day in range(7):    
            df_dayweek = df_train_valid.loc[df_train_valid["Day of Week"]==day]
            # if some day of week is missing, we use the participant level mean
            if len(df_dayweek) == 0:
                dayweek_level_mean[day] = df_train_valid.loc[df_train_valid["step_mask"]==1, "step_rate"].mean()
            else:
                dayweek_level_mean[day] = df_dayweek.loc[df_dayweek["step_mask"]==1, "step_rate"].mean()

        # test
        df_test["step_rate_pred"] = df_test.apply(lambda x: dayweek_level_mean[x["Day of Week"]], axis=1)
        df_test["steps_pred"] = df_test["step_rate_pred"] * df_test["valid_minutes"]
        test_mae = np.mean(np.abs(df_test["steps"].values - df_test["steps_pred"].values))
        test_mse = np.mean((df_test["steps"].values - df_test["steps_pred"].values)**2)

        # record the performance on the test set
        test_loss["mae"].append(test_mae)
        test_loss["mse"].append(test_mse)
        test_loss["rmse"].append(np.sqrt(test_mse))

        # also record the number of test hourly blocks
        test_loss["num_hourly_blocks"].append(len(df_test))
        
        if verbose:
            print(f"split {i} | test mae: {test_mae:.2f} | test mse: {test_mse:.2f} | test rmse: {np.sqrt(test_mse):.2f}")

    # record the pid
    test_loss["participant_id"] = df_exp["Participant ID"].unique().item()
            
    if verbose:
        print()
        print(f'mean(std) | test mae: {np.mean(test_loss["mae"]):.2f}({np.std(test_loss["mae"]):.2f})')
        print(f'mean(std) | test mse: {np.mean(test_loss["mse"]):.2f}({np.std(test_loss["mse"]):.2f})')
        print(f'mean(std) | test rmse: {np.mean(test_loss["rmse"]):.2f}({np.std(test_loss["rmse"]):.2f})')

    return test_loss

def hour_mean(df_exp, num_splits=10, verbose=False):
    """
    Fill with hour of the day mean (mean of the all valid hourly blocks)
    Args:
        - df_exp: the dataframe of the participant
        - num_splits: number of splits in the data
        - verbose: wheter to print the metrics we computed for each split of this participant
    """
    
    # construct the dictionary to record and test performance
    test_loss = {"mae": [], "mse": [], "rmse":[], "num_hourly_blocks": []}
    
    if verbose:
        print(f"participant id = {df_exp['Participant ID'].unique().item()}")
        print("-" * 100)
    
    for i in range(num_splits):
        # split the dataframe into train, valid and test
        # Note that train, valid and test accounts for all the valid hourly blocks between 6:00 ~ 22:00
        df_train_valid = df_exp.loc[(df_exp[f"train_mask_split{i}"]==1)|(df_exp[f"valid_mask_split{i}"]==1)]
        df_test = df_exp.loc[df_exp[f"test_mask_split{i}"]==1]

        # train
        hour_level_mean = {}
        for hour in range(24):
            df_hour = df_train_valid.loc[df_train_valid["Hour of Day"]==hour]
            if len(df_hour) == 0:
                hour_level_mean[hour] = df_train_valid.loc[df_train_valid["step_mask"]==1, "step_rate"].mean()
            else:
                hour_level_mean[hour] = df_hour.loc[df_hour["step_mask"]==1, "step_rate"].mean()

        # test
        df_test["step_rate_pred"] = df_test.apply(lambda x: hour_level_mean[x["Hour of Day"]], axis=1)
        df_test["steps_pred"] = df_test["step_rate_pred"] * df_test["valid_minutes"]
        test_mae = np.mean(np.abs(df_test["steps"].values - df_test["steps_pred"].values))
        test_mse = np.mean((df_test["steps"].values - df_test["steps_pred"].values)**2)

        # record the performance on the test set
        test_loss["mae"].append(test_mae)
        test_loss["mse"].append(test_mse)
        test_loss["rmse"].append(np.sqrt(test_mse))

        # also record the number of test hourly blocks
        test_loss["num_hourly_blocks"].append(len(df_test))

        if verbose:
            print(f"split {i} | test mae: {test_mae:.2f} | test mse: {test_mse:.2f} | test rmse: {np.sqrt(test_mse):.2f}")

    # record the pid
    test_loss["participant_id"] = df_exp["Participant ID"].unique().item()
            
    if verbose:
        print()
        print(f'mean(std) | test mae: {np.mean(test_loss["mae"]):.2f}({np.std(test_loss["mae"]):.2f})')
        print(f'mean(std) | test mse: {np.mean(test_loss["mse"]):.2f}({np.std(test_loss["mse"]):.2f})')
        print(f'mean(std) | test rmse: {np.mean(test_loss["rmse"]):.2f}({np.std(test_loss["rmse"]):.2f})')

    return test_loss

def dayweek_hour_mean(df_exp, num_splits=10, verbose=False):
    """
    Fill with day of the week and hour of the day mean (mean of the all valid hourly blocks)
    Args:
        - df_exp: the dataframe of the participant
        - num_splits: number of splits in the data
        - verbose: wheter to print the metrics we computed for each split of this participant
    """
    
    # construct the dictionary to record and test performance
    test_loss = {"mae": [], "mse": [], "rmse":[], "num_hourly_blocks": []}
    
    if verbose:
        print(f"participant id = {df_exp['Participant ID'].unique().item()}")
        print("-" * 100)
    
    for i in range(num_splits):
        # split the dataframe into train, valid and test
        # Note that train, valid and test accounts for all the valid hourly blocks between 6:00 ~ 22:00
        df_train_valid = df_exp.loc[(df_exp[f"train_mask_split{i}"]==1)|(df_exp[f"valid_mask_split{i}"]==1)]
        df_test = df_exp.loc[df_exp[f"test_mask_split{i}"]==1]

        # train
        dayweek_hourly_mean = {}
        for day in range(7):
            dayweek_hourly_mean[day] = {}
            df_dayweek = df_train_valid.loc[df_train_valid["Day of Week"]==day]
            for hour in range(24):
                df_dayweek_hour = df_dayweek.loc[df_dayweek["Hour of Day"]==hour]
                if len(df_dayweek_hour) == 0:
                    dayweek_hourly_mean[day][hour] = df_train_valid.loc[df_train_valid["step_mask"]==1, "step_rate"].mean()
                else:
                    dayweek_hourly_mean[day][hour] = df_dayweek_hour.loc[df_dayweek_hour["step_mask"]==1, "step_rate"].mean()

        # test
        df_test["step_rate_pred"] = df_test.apply(lambda x: dayweek_hourly_mean[x["Day of Week"]][x["Hour of Day"]], axis=1)
        df_test["steps_pred"] = df_test["step_rate_pred"] * df_test["valid_minutes"]
        test_mae = np.mean(np.abs(df_test["steps"].values - df_test["steps_pred"].values))
        test_mse = np.mean((df_test["steps"].values - df_test["steps_pred"].values)**2)

        # record the performance on the test set
        test_loss["mae"].append(test_mae)
        test_loss["mse"].append(test_mse)
        test_loss["rmse"].append(np.sqrt(test_mse))

        # also record the number of test hourly blocks
        test_loss["num_hourly_blocks"].append(len(df_test))

        if verbose:
            print(f"split {i} | test mae: {test_mae:.2f} | test mse: {test_mse:.2f} | test rmse: {np.sqrt(test_mse):.2f}")

    # record the pid
    test_loss["participant_id"] = df_exp["Participant ID"].unique().item()
            
    if verbose:
        print()
        print(f'mean(std) | test mae: {np.mean(test_loss["mae"]):.2f}({np.std(test_loss["mae"]):.2f})')
        print(f'mean(std) | test mse: {np.mean(test_loss["mse"]):.2f}({np.std(test_loss["mse"]):.2f})')
        print(f'mean(std) | test rmse: {np.mean(test_loss["rmse"]):.2f}({np.std(test_loss["rmse"]):.2f})')

    return test_loss

def participant_median(df_exp, num_splits=10, verbose=False):
    """
    Fill with the median of all of the valid hourly blocks in the participant data. Since we only care about 
    6:00 ~ 22:00, we only include 6:00 ~ 22:00 data to compute the median
    Args:
        - df_exp: the dataframe of the participant
        - num_splits: number of splits in the data
        - verbose: wheter to print the metrics we computed for each split of this participant
    """
    
    # construct the dictionary to record and test performance
    test_loss = {"mae": [], "mse": [], "rmse":[], "num_hourly_blocks": []}
    
    if verbose:
        print(f"participant id = {df_exp['Participant ID'].unique().item()}")
        print("-" * 100)
    
    for i in range(num_splits):
        # split the dataframe into train, valid and test
        # Note that train, valid and test accounts for all the valid hourly blocks between 6:00 ~ 22:00
        df_train_valid = df_exp.loc[(df_exp[f"train_mask_split{i}"]==1)|(df_exp[f"valid_mask_split{i}"]==1)]
        df_test = df_exp.loc[df_exp[f"test_mask_split{i}"]==1]

        # train
        part_step_rate_median = df_train_valid.loc[df_train_valid["step_mask"]==1, "step_rate"].median()

        # test
        df_test["step_rate_pred"] = part_step_rate_median
        df_test["steps_pred"] = df_test["step_rate_pred"] * df_test["valid_minutes"]
        test_mae = np.mean(np.abs(df_test["steps"].values - df_test["steps_pred"].values))
        test_mse = np.mean((df_test["steps"].values - df_test["steps_pred"].values)**2)
        
        # record the performance on the test set
        test_loss["mae"].append(test_mae)
        test_loss["mse"].append(test_mse)
        test_loss["rmse"].append(np.sqrt(test_mse))

        # also record the number of test hourly blocks
        test_loss["num_hourly_blocks"].append(len(df_test))
        
        if verbose:
            print(f"split {i} | test mae: {test_mae:.2f} | test mse: {test_mse:.2f} | test rmse: {np.sqrt(test_mse):.2f}")

    # record the pid
    test_loss["participant_id"] = df_exp["Participant ID"].unique().item()
            
    if verbose:
        print()
        print(f'mean(std) | test mae: {np.mean(test_loss["mae"]):.2f}({np.std(test_loss["mae"]):.2f})')
        print(f'mean(std) | test mse: {np.mean(test_loss["mse"]):.2f}({np.std(test_loss["mse"]):.2f})')
        print(f'mean(std) | test rmse: {np.mean(test_loss["rmse"]):.2f}({np.std(test_loss["rmse"]):.2f})')

    return test_loss

def dayweek_median(df_exp, num_splits=10, verbose=False):
    """
    Fill with day of the week median (median of the all valid hourly blocks)
    Args:
        - df_exp: the dataframe of the participant
        - num_splits: number of splits in the data
        - verbose: wheter to print the metrics we computed for each split of this participant
    """
    
    # construct the dictionary to record and test performance
    test_loss = {"mae": [], "mse": [], "rmse":[], "num_hourly_blocks": []}
    
    if verbose:
        print(f"participant id = {df_exp['Participant ID'].unique().item()}")
        print("-" * 100)
    
    for i in range(num_splits):
        # split the dataframe into train, valid and test
        # Note that train, valid and test accounts for all the valid hourly blocks between 6:00 ~ 22:00
        df_train_valid = df_exp.loc[(df_exp[f"train_mask_split{i}"]==1)|(df_exp[f"valid_mask_split{i}"]==1)]
        df_test = df_exp.loc[df_exp[f"test_mask_split{i}"]==1]

        # train
        dayweek_level_median = {}
        for day in range(7):    
            df_dayweek = df_train_valid.loc[df_train_valid["Day of Week"]==day]
            # if some day of week is missing, we use the participant level mean
            if len(df_dayweek) == 0:
                dayweek_level_median[day] = df_train_valid.loc[df_train_valid["step_mask"]==1, "step_rate"].median()
            else:
                dayweek_level_median[day] = df_dayweek.loc[df_dayweek["step_mask"]==1, "step_rate"].median()

        # test
        df_test["step_rate_pred"] = df_test.apply(lambda x: dayweek_level_median[x["Day of Week"]], axis=1)
        df_test["steps_pred"] = df_test["step_rate_pred"] * df_test["valid_minutes"]
        test_mae = np.mean(np.abs(df_test["steps"].values - df_test["steps_pred"].values))
        test_mse = np.mean((df_test["steps"].values - df_test["steps_pred"].values)**2)

        # record the performance on the test set
        test_loss["mae"].append(test_mae)
        test_loss["mse"].append(test_mse)
        test_loss["rmse"].append(np.sqrt(test_mse))

        # also record the number of test hourly blocks
        test_loss["num_hourly_blocks"].append(len(df_test))
        
        if verbose:
            print(f"split {i} | test mae: {test_mae:.2f} | test mse: {test_mse:.2f} | test rmse: {np.sqrt(test_mse):.2f}")

    # record the pid
    test_loss["participant_id"] = df_exp["Participant ID"].unique().item()
            
    if verbose:
        print()
        print(f'mean(std) | test mae: {np.mean(test_loss["mae"]):.2f}({np.std(test_loss["mae"]):.2f})')
        print(f'mean(std) | test mse: {np.mean(test_loss["mse"]):.2f}({np.std(test_loss["mse"]):.2f})')
        print(f'mean(std) | test rmse: {np.mean(test_loss["rmse"]):.2f}({np.std(test_loss["rmse"]):.2f})')

    return test_loss

def hour_median(df_exp, num_splits=10, verbose=False):
    """
    Fill with hour of the day median (median of the all valid hourly blocks)
    Args:
        - df_exp: the dataframe of the participant
        - num_splits: number of splits in the data
        - verbose: wheter to print the metrics we computed for each split of this participant
    """
    
    # construct the dictionary to record and test performance
    test_loss = {"mae": [], "mse": [], "rmse":[], "num_hourly_blocks": []}
    
    if verbose:
        print(f"participant id = {df_exp['Participant ID'].unique().item()}")
        print("-" * 100)
    
    for i in range(num_splits):
        # split the dataframe into train, valid and test
        # Note that train, valid and test accounts for all the valid hourly blocks between 6:00 ~ 22:00
        df_train_valid = df_exp.loc[(df_exp[f"train_mask_split{i}"]==1)|(df_exp[f"valid_mask_split{i}"]==1)]
        df_test = df_exp.loc[df_exp[f"test_mask_split{i}"]==1]

        # train
        hour_level_median = {}
        for hour in range(24):
            df_hour = df_train_valid.loc[df_train_valid["Hour of Day"]==hour]
            if len(df_hour) == 0:
                hour_level_median[hour] = df_train_valid.loc[df_train_valid["step_mask"]==1, "step_rate"].median()
            else:
                hour_level_median[hour] = df_hour.loc[df_hour["step_mask"]==1, "step_rate"].median()

        # test
        df_test["step_rate_pred"] = df_test.apply(lambda x: hour_level_median[x["Hour of Day"]], axis=1)
        df_test["steps_pred"] = df_test["step_rate_pred"] * df_test["valid_minutes"]
        test_mae = np.mean(np.abs(df_test["steps"].values - df_test["steps_pred"].values))
        test_mse = np.mean((df_test["steps"].values - df_test["steps_pred"].values)**2)

        # record the performance on the test set
        test_loss["mae"].append(test_mae)
        test_loss["mse"].append(test_mse)
        test_loss["rmse"].append(np.sqrt(test_mse))

        # also record the number of test hourly blocks
        test_loss["num_hourly_blocks"].append(len(df_test))

        if verbose:
            print(f"split {i} | test mae: {test_mae:.2f} | test mse: {test_mse:.2f} | test rmse: {np.sqrt(test_mse):.2f}")

    # record the pid
    test_loss["participant_id"] = df_exp["Participant ID"].unique().item()
            
    if verbose:
        print()
        print(f'mean(std) | test mae: {np.mean(test_loss["mae"]):.2f}({np.std(test_loss["mae"]):.2f})')
        print(f'mean(std) | test mse: {np.mean(test_loss["mse"]):.2f}({np.std(test_loss["mse"]):.2f})')
        print(f'mean(std) | test rmse: {np.mean(test_loss["rmse"]):.2f}({np.std(test_loss["rmse"]):.2f})')

    return test_loss

def dayweek_hour_median(df_exp, num_splits=10, verbose=False):
    """
    Fill with day of the week and hour of the day median (median of the all valid hourly blocks)
    Args:
        - df_exp: the dataframe of the participant
        - num_splits: number of splits in the data
        - verbose: wheter to print the metrics we computed for each split of this participant
    """
    
    # construct the dictionary to record and test performance
    test_loss = {"mae": [], "mse": [], "rmse":[], "num_hourly_blocks": []}
    
    if verbose:
        print(f"participant id = {df_exp['Participant ID'].unique().item()}")
        print("-" * 100)
    
    for i in range(num_splits):
        # split the dataframe into train, valid and test
        # Note that train, valid and test accounts for all the valid hourly blocks between 6:00 ~ 22:00
        df_train_valid = df_exp.loc[(df_exp[f"train_mask_split{i}"]==1)|(df_exp[f"valid_mask_split{i}"]==1)]
        df_test = df_exp.loc[df_exp[f"test_mask_split{i}"]==1]

        # train
        dayweek_hourly_median = {}
        for day in range(7):
            dayweek_hourly_median[day] = {}
            df_dayweek = df_train_valid.loc[df_train_valid["Day of Week"]==day]
            for hour in range(24):
                df_dayweek_hour = df_dayweek.loc[df_dayweek["Hour of Day"]==hour]
                if len(df_dayweek_hour) == 0:
                    dayweek_hourly_median[day][hour] = df_train_valid.loc[df_train_valid["step_mask"]==1, "step_rate"].median()
                else:
                    dayweek_hourly_median[day][hour] = df_dayweek_hour.loc[df_dayweek_hour["step_mask"]==1, "step_rate"].median()

        # test
        df_test["step_rate_pred"] = df_test.apply(lambda x: dayweek_hourly_median[x["Day of Week"]][x["Hour of Day"]], axis=1)
        df_test["steps_pred"] = df_test["step_rate_pred"] * df_test["valid_minutes"]
        test_mae = np.mean(np.abs(df_test["steps"].values - df_test["steps_pred"].values))
        test_mse = np.mean((df_test["steps"].values - df_test["steps_pred"].values)**2)

        # # write it intot he file
        # pred_bundles = {"true": df_test["steps"].values, "pred": df_test["steps_pred"].values}
        # pid = df_exp["Participant ID"].unique().item()
        # with open(f"{data_utils.FILE_CACHE}/dayweek_hour_median_pid_{pid}_split_{i}.pkl", "wb") as fout:
        #     pickle.dump(pred_bundles, fout)

        # record the performance on the test set
        test_loss["mae"].append(test_mae)
        test_loss["mse"].append(test_mse)
        test_loss["rmse"].append(np.sqrt(test_mse))

        # also record the number of test hourly blocks
        test_loss["num_hourly_blocks"].append(len(df_test))

        if verbose:
            print(f"split {i} | test mae: {test_mae:.2f} | test mse: {test_mse:.2f} | test rmse: {np.sqrt(test_mse):.2f}")

    # record the pid
    test_loss["participant_id"] = df_exp["Participant ID"].unique().item()
            
    if verbose:
        print()
        print(f'mean(std) | test mae: {np.mean(test_loss["mae"]):.2f}({np.std(test_loss["mae"]):.2f})')
        print(f'mean(std) | test mse: {np.mean(test_loss["mse"]):.2f}({np.std(test_loss["mse"]):.2f})')
        print(f'mean(std) | test rmse: {np.mean(test_loss["rmse"]):.2f}({np.std(test_loss["rmse"]):.2f})')

    return test_loss

def forward_filling(df_exp, num_splits=10, verbose=False):
    # construct the dictionary to record the test performance for forward and backward filling
    test_loss = {"mae": [], "mse": [], "rmse": [], "num_hourly_blocks": []}
    
    if verbose:
        print(f"participant id = {df_exp['Participant ID'].unique().item()}")
        print("-" * 100)
    
    for i in range(num_splits):
        df_exp_fd = df_exp.copy(deep=True)
        df_exp_fd = df_exp_fd[["Participant ID", "steps", "valid_minutes", 
                                 "step_mask", "step_rate", f"train_mask_split{i}", f"valid_mask_split{i}",
                                 f"test_mask_split{i}"]]

        # step 1: set all the step rate at the positions of step_mask==0 as np.nan
        # since in the test set, the only thing is missing is the current block and all the originally missing blocks
        df_exp_fd.loc[df_exp_fd["step_mask"]==0, "step_rate"] = np.nan 

        # step 2: copy the step_rate column into step_rate_ffill and step_rate_bfill
        df_exp_fd["step_rate_ffill"] = df_exp_fd["step_rate"]

        # step 3: forward fill
        df_exp_fd["step_rate_ffill"] = df_exp_fd["step_rate_ffill"].fillna(method="ffill")
        df_exp_fd["step_rate_ffill"] = df_exp_fd["step_rate_ffill"].shift(1)

        # step 4: fill the missing data in the step_rate_ffill and step_rate_bfill using the participant median
        # note this is not suitable for computing the average of back and foward fill
        # over there, we don't use any value to fill the missing value if any method can have valid value over there
        # if both are missing, then we use the participant level median to fill
        df_train_valid = df_exp_fd.loc[(df_exp_fd[f"train_mask_split{i}"]==1)|(df_exp_fd[f"valid_mask_split{i}"]==1)]
        part_step_rate_median = df_train_valid.loc[df_train_valid["step_mask"]==1, "step_rate"].median()
        df_exp_fd["step_rate_ffill"].fillna(part_step_rate_median, inplace=True)

        assert df_exp_fd["step_rate_ffill"].notna().all(), "ffill has NaN!"

        # step 5: make the prediction
        df_exp_fd_test = df_exp_fd.loc[df_exp_fd[f"test_mask_split{i}"]==1]
        df_exp_fd_test["pred_ffill"] = df_exp_fd_test["step_rate_ffill"] * df_exp_fd_test["valid_minutes"]

        # step 6: compute the mae
        mae_ffill = np.mean(np.abs(df_exp_fd_test["steps"].values - df_exp_fd_test["pred_ffill"].values))
        test_loss["mae"].append(mae_ffill)

        # step 7: compute the mse
        mse_ffill = np.mean((df_exp_fd_test["steps"].values - df_exp_fd_test["pred_ffill"].values)**2)
        test_loss["mse"].append(mse_ffill)
        
        # step 7: compute the rmse
        rmse_ffill = np.sqrt(np.mean((df_exp_fd_test["steps"].values - df_exp_fd_test["pred_ffill"].values)**2))
        test_loss["rmse"].append(rmse_ffill)

        # also record the number of test hourly blocks
        test_loss["num_hourly_blocks"].append(len(df_exp_fd_test))

        if verbose:
            print(f"split {i} | test mae: {mae_ffill:.2f} | test mse: {mse_ffill:.2f} | test rmse: {rmse_ffill:.2f}")

    # record the pid
    test_loss["participant_id"] = df_exp["Participant ID"].unique().item()
            
    if verbose:
        print()
        print(f'mean(std) | test mae: {np.mean(test_loss["mae"]):.2f}({np.std(test_loss["mae"]):.2f})')
        print(f'mean(std) | test mse: {np.mean(test_loss["mse"]):.2f}({np.std(test_loss["mse"]):.2f})')
        print(f'mean(std) | test rmse: {np.mean(test_loss["rmse"]):.2f}({np.std(test_loss["rmse"]):.2f})')

    return test_loss

def backward_filling(df_exp, num_splits=10, verbose=False):
    # construct the dictionary to record the test performance for backward filling
    test_loss = {"mae": [], "mse": [], "rmse": [], "num_hourly_blocks": []}
    
    if verbose:
        print(f"participant id = {df_exp['Participant ID'].unique().item()}")
        print("-" * 100)
    
    for i in range(num_splits):
        df_exp_bd = df_exp.copy(deep=True)
        df_exp_bd = df_exp_bd[["Participant ID", "steps", "valid_minutes", 
                                 "step_mask", "step_rate", f"train_mask_split{i}", f"valid_mask_split{i}",
                                 f"test_mask_split{i}"]]

        # step 1: set all the step rate at the positions of step_mask==0 as np.nan
        # since in the test set, the only thing is missing is the current block and all the originally missing blocks
        df_exp_bd.loc[df_exp_bd["step_mask"]==0, "step_rate"] = np.nan 

        # step 2: copy the step_rate column into and step_rate_bfill
        df_exp_bd["step_rate_bfill"] = df_exp_bd["step_rate"]

        # step 3: backward fill
        df_exp_bd["step_rate_bfill"] = df_exp_bd["step_rate_bfill"].fillna(method="bfill")
        df_exp_bd["step_rate_bfill"] = df_exp_bd["step_rate_bfill"].shift(-1)

        # step 4: fill the missing data in the step_rate_ffill and step_rate_bfill using the participant median
        # note this is not suitable for computing the average of back and foward fill
        # over there, we don't use any value to fill the missing value if any method can have valid value over there
        # if both are missing, then we use the participant level median to fill
        df_train_valid = df_exp_bd.loc[(df_exp_bd[f"train_mask_split{i}"]==1)|(df_exp_bd[f"valid_mask_split{i}"]==1)]
        part_step_rate_median = df_train_valid.loc[df_train_valid["step_mask"]==1, "step_rate"].median()
        df_exp_bd["step_rate_bfill"].fillna(part_step_rate_median, inplace=True)

        assert df_exp_bd["step_rate_bfill"].notna().all(), "bfill has NaN!"

        # step 5: make the prediction
        df_exp_bd_test = df_exp_bd.loc[df_exp_bd[f"test_mask_split{i}"]==1]
        df_exp_bd_test["pred_bfill"] = df_exp_bd_test["step_rate_bfill"] * df_exp_bd_test["valid_minutes"]

        # step 6: compute the mae
        mae_bfill = np.mean(np.abs(df_exp_bd_test["steps"].values - df_exp_bd_test["pred_bfill"].values))
        test_loss["mae"].append(mae_bfill)

        # step 7: compute the mae
        mse_bfill = np.mean((df_exp_bd_test["steps"].values - df_exp_bd_test["pred_bfill"].values)**2)
        test_loss["mse"].append(mse_bfill)
        
        # step 7: compute the rmse
        rmse_bfill = np.sqrt(np.mean((df_exp_bd_test["steps"].values - df_exp_bd_test["pred_bfill"].values)**2))
        test_loss["rmse"].append(rmse_bfill)

        # also record the number of test hourly blocks
        test_loss["num_hourly_blocks"].append(len(df_exp_bd_test))

        if verbose:
            print(f"split {i} | test mae: {mae_bfill:.2f} | test mse: {mse_bfill:.2f} | test rmse: {rmse_bfill:.2f}")

    # record the pid
    test_loss["participant_id"] = df_exp["Participant ID"].unique().item()
            
    if verbose:
        print()
        print(f'mean(std) | test mae: {np.mean(test_loss["mae"]):.2f}({np.std(test_loss["mae"]):.2f})')
        print(f'mean(std) | test mse: {np.mean(test_loss["mse"]):.2f}({np.std(test_loss["mse"]):.2f})')
        print(f'mean(std) | test rmse: {np.mean(test_loss["rmse"]):.2f}({np.std(test_loss["rmse"]):.2f})')

    return test_loss

def avg_ffill_bfill(row, part_level_mean):
    if (not np.isnan(row["step_rate_ffill"])) and (not np.isnan(row["step_rate_bfill"])):
        return (row["step_rate_ffill"] + row["step_rate_bfill"]) / 2
    elif np.isnan(row["step_rate_ffill"]) and (not np.isnan(row["step_rate_bfill"])):
        return row["step_rate_bfill"]
    elif (not np.isnan(row["step_rate_ffill"])) and np.isnan(row["step_rate_bfill"]):
        return row["step_rate_ffill"]
    else:
        return part_level_mean
    
def avg_forward_backward_fill(df_exp, num_splits=10, verbose=False):
    # construct the dictionary to record the test performance for forward and backward filling
    test_loss = {"mae": [], "mse": [], "rmse": [], "num_hourly_blocks": []}
    
    if verbose:
        print(f"participant id = {df_exp['Participant ID'].unique().item()}")
        print("-" * 100)
    
    # average of forward and backward filling
    for split_idx in range(num_splits):
        df_exp_fb = df_exp.copy(deep=True)
        df_exp_fb = df_exp_fb[["Participant ID", "steps", "valid_minutes", 
                                 "step_mask", "step_rate", f"train_mask_split{split_idx}", f"valid_mask_split{split_idx}",
                                 f"test_mask_split{split_idx}"]]

        # step 1: set all the step rate at the positions of step_mask==0 as np.nan
        # since in the test set, the only thing is missing is the current block and all the originally missing blocks
        df_exp_fb.loc[df_exp_fb["step_mask"]==0, "step_rate"] = np.nan 

        # step 2: copy the step_rate column into step_rate_ffill and step_rate_bfill
        df_exp_fb["step_rate_ffill"] = df_exp_fb["step_rate"]
        df_exp_fb["step_rate_bfill"] = df_exp_fb["step_rate"]

        # step 3: forward fill and backward fill 
        df_exp_fb["step_rate_ffill"] = df_exp_fb["step_rate_ffill"].fillna(method="ffill")
        df_exp_fb["step_rate_ffill"] = df_exp_fb["step_rate_ffill"].shift(1)

        df_exp_fb["step_rate_bfill"] = df_exp_fb["step_rate_bfill"].fillna(method="bfill")
        df_exp_fb["step_rate_bfill"] = df_exp_fb["step_rate_bfill"].shift(-1)

        # step 4: fill the missing data in the step_rate_ffill and step_rate_bfill using the participant median
        # note this is not suitable for computing the average of back and foward fill
        # over there, we don't use any value to fill the missing value if any method can have valid value over there
        # if both are missing, then we use the participant level median to fill
        df_train_valid = df_exp_fb.loc[(df_exp_fb[f"train_mask_split{split_idx}"]==1)|(df_exp_fb[f"valid_mask_split{split_idx}"]==1)]
        part_step_rate_median = df_train_valid.loc[df_train_valid["step_mask"]==1, "step_rate"].median()
        df_exp_fb["step_rate_fbfill"] = df_exp_fb.apply(lambda row: avg_ffill_bfill(row, part_step_rate_median), axis=1)

        assert df_exp_fb["step_rate_fbfill"].notna().all(), "fbfill has NaN!"

        # step 5: make the prediction
        df_exp_fb_test = df_exp_fb.loc[df_exp_fb[f"test_mask_split{split_idx}"]==1]
        df_exp_fb_test["pred_fbfill"] = df_exp_fb_test["step_rate_fbfill"] * df_exp_fb_test["valid_minutes"]

        # step 6: compute the mae
        mae_fbfill = np.mean(np.abs(df_exp_fb_test["steps"].values - df_exp_fb_test["pred_fbfill"].values))
        test_loss["mae"].append(mae_fbfill)
        
        # step 7: compute the mse
        mse_fbfill = np.mean((df_exp_fb_test["steps"].values - df_exp_fb_test["pred_fbfill"].values)**2)
        test_loss["mse"].append(mse_fbfill)

        # step 8: compute the rmse
        rmse_fbfill = np.sqrt(np.mean((df_exp_fb_test["steps"].values - df_exp_fb_test["pred_fbfill"].values)**2))
        test_loss["rmse"].append(rmse_fbfill)

        # also record the number of test hourly blocks
        test_loss["num_hourly_blocks"].append(len(df_exp_fb_test))

        if verbose:
            print(f"split {split_idx} | test mae: {mae_fbfill:.2f} | test mse: {mse_fbfill:.2f} | test rmse: {rmse_fbfill:.2f}")

    # record the pid
    test_loss["participant_id"] = df_exp["Participant ID"].unique().item()
            
    if verbose:
        print()
        print(f'mean(std) | test mae: {np.mean(test_loss["mae"]):.2f}({np.std(test_loss["mae"]):.2f})')
        print(f'mean(std) | test mse: {np.mean(test_loss["mse"]):.2f}({np.std(test_loss["mse"]):.2f})')
        print(f'mean(std) | test rmse: {np.mean(test_loss["rmse"]):.2f}({np.std(test_loss["rmse"]):.2f})')

    return test_loss

def zero_fill(df_exp, num_splits=10, verbose=False):
    """
    Fill with the zeros. 
    Args:
        - df_exp: the dataframe of the participant
        - num_splits: number of splits in the data
        - verbose: wheter to print the metrics we computed for each split of this participant
    """
    
    # construct the dictionary to record and test performance
    test_loss = {"mae": [], "mse": [], "rmse":[], "num_hourly_blocks": []}
    
    if verbose:
        print(f"participant id = {df_exp['Participant ID'].unique().item()}")
        print("-" * 100)
    
    for i in range(num_splits):
        # split the dataframe into train, valid and test
        # Note that train, valid and test accounts for all the valid hourly blocks between 6:00 ~ 22:00
        df_train_valid = df_exp.loc[(df_exp[f"train_mask_split{i}"]==1)|(df_exp[f"valid_mask_split{i}"]==1)]
        df_test = df_exp.loc[df_exp[f"test_mask_split{i}"]==1]

        # test
        df_test["step_rate_pred"] = 0.0
        df_test["steps_pred"] = df_test["step_rate_pred"] * df_test["valid_minutes"]
        test_mae = np.mean(np.abs(df_test["steps"].values - df_test["steps_pred"].values))
        test_mse = np.mean((df_test["steps"].values - df_test["steps_pred"].values)**2)
        
        # record the performance on the test set
        test_loss["mae"].append(test_mae)
        test_loss["mse"].append(test_mse)
        test_loss["rmse"].append(np.sqrt(test_mse))

        # also record the number of test hourly blocks
        test_loss["num_hourly_blocks"].append(len(df_test))
        
        if verbose:
            print(f"split {i} | test mae: {test_mae:.2f} | test mse: {test_mse:.2f} | test rmse: {np.sqrt(test_mse):.2f}")

    # record the pid
    test_loss["participant_id"] = df_exp["Participant ID"].unique().item()
            
    if verbose:
        print()
        print(f'mean(std) | test mae: {np.mean(test_loss["mae"]):.2f}({np.std(test_loss["mae"]):.2f})')
        print(f'mean(std) | test mse: {np.mean(test_loss["mse"]):.2f}({np.std(test_loss["mse"]):.2f})')
        print(f'mean(std) | test rmse: {np.mean(test_loss["rmse"]):.2f}({np.std(test_loss["rmse"]):.2f})')

    return test_loss


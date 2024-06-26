"""
This script contains all the utility functions for external validation on the held-out test set
"""
from sklearn.model_selection import train_test_split
import os
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import subprocess
import datetime
from tqdm import tqdm
import pickle
from google.cloud import bigquery

import sys
sys.path.append("../..")

from utils.data_utils import pull_file, get_data_by_id, normalize_data, FILE_CACHE

import warnings
warnings.filterwarnings("ignore")
np.set_printoptions(suppress=True)

HIGH_MISS_START_END_FILE = "df_cohort_extvalid_high_missrate.parquet"
HIGH_MISS_SHIFT_FILE = "dayweek_shift_extvalid_high_missrate.parquet"

LOW_MISS_START_END_FILE = "df_cohort_extvalid_low_missrate.parquet"
LOW_MISS_SHIFT_FILE = "dayweek_shift_extvalid_low_missrate.parquet"

## different from data_utils.get_hourly_data, for the external validation, 
## we treat every observed hourly block as test 
## and we dont have multiple splits anymore
def get_hourly_data(pid, df_start_end_day, df_best_shift_dw, start_hour=6, end_hour=22, conv_feat=True, return_time_dict=False):
    """
    Get the hourly level data after preprocessing, the output can be used to initialize the Dataset class.
    We can choose which period (start_hour to end_hour) we would like to include in the train/valid/test set.
    Args:
        - pid: Participant ID
        - df_start_end_day: pandas dataframe containing the start and end day for each cropped qualified subsequence 
        - df_best_shift_dw: pandas dataframe containing the best shift for day of the week
        - start_hour: the start hour included in the train/valid/test set
        - end_hour: the end hour included in the train/valid/test set
        - conv_feat: bool, whether to create the features for the convolution-based model.
        - return_time_dict: bool, whether to return the dictionary which contains train/valid/test split information based on step count bins
    """

    # get the dataframe for the particular pid
    df_exp = get_data_by_id(pid)
    assert df_exp.index.levels[0][0] == pid, f"wrong participant data {df_exp.index.levels[0][0]} is read in!"

    ### preprocess ###
    # put participant id and datetime into the columns and reset the index
    df_exp.reset_index(inplace=True)

    # get the start and end day from the precomputed file
    #df_start_end_day = pd.read_parquet(f"{FILE_CACHE}/{HIGH_MISS_START_END_FILE}")
    start_day, end_day = df_start_end_day.loc[pid, ["Start Day", "End Day"]]
    df_exp = df_exp.loc[(df_exp["Study day"]>=start_day) & (df_exp["Study day"]<=end_day)]

    # step 3: reset the index since we remove the first and last several days which have no valid minutes
    df_exp.reset_index(drop=True, inplace=True)

    # get the date and time columns
    df_exp["date"] = df_exp["datetime"].dt.date
    df_exp["time"] = df_exp["datetime"].dt.time
    
    # set steps to be np.nan when valid_minutes equal to zero
    df_exp.loc[df_exp["valid_minutes"]==0, "steps"] = np.nan
    # set heart rate to be np.nan when valid_minutes equal to zero
    df_exp.loc[df_exp["valid_minutes"]==0, "heart_rate"] = np.nan

    # add the step rate column
    df_exp["step_rate"] = df_exp["steps"] / df_exp["valid_minutes"]
    # reset the study date to make the first valid study day as zero
    first_valid_study_day = df_exp.iloc[0]["Study day"]
    df_exp["Study day"] = df_exp["Study day"] - first_valid_study_day
    # normalize the data and get the mean and std of the step rates
    df_exp, step_rate_mean, step_rate_std = normalize_data(df_exp)
    
    # check if all the hours are available for the period
    timediff = (df_exp.iloc[-1].datetime - df_exp.iloc[0].datetime)
    assert timediff / datetime.timedelta(hours=1) == len(df_exp)-1, "some hours are missing"
    # second check: manually compute the hour
    def compute_time_axis(row):
        return row["Study day"]*24 + row["Hour of Day"]
    df_exp["time_axis"] = df_exp.apply(lambda x: compute_time_axis(x), axis=1)
    # make the first hour as the start hour which is zero
    df_exp["time_axis"] = df_exp["time_axis"] - df_exp.iloc[0]["time_axis"]
    assert (df_exp["time_axis"] == df_exp.index).all(), "some hours are missing"

    ### Feature Engineering ###    
    # shift the day of the week
    best_shift = df_best_shift_dw.loc[pid, "Best Shift"]
    df_exp["Day of Week"] = (df_exp["Day of Week"] + best_shift) % 7 
    df_exp["Is Weekend Day"] = (df_exp["Day of Week"].isin([5, 6])).astype("int")

    # add the step rate missingness indicator ###
    df_exp["step_mask"] = (df_exp["valid_minutes"]>0).astype("int")
    # fill the missing values in the step counts and step rates
    df_exp["steps"].fillna(value=0.0, inplace=True)
    df_exp["step_rate"].fillna(value=0.0, inplace=True)
    df_exp["step_rate_norm"].fillna(value=0.0, inplace=True)
    # fill the missing values in the heart rate
    df_exp["heart_rate"].fillna(value=0.0, inplace=True)
    df_exp["heart_rate_norm"].fillna(value=0.0, inplace=True)

    assert df_exp.notnull().all().all(), f"there is still NaN after filling heart rate and step rate of pid={pid}"

    ### Here, we don't need to split the data into multiple splits ###
    ### We don't have train/valid/test split as well ###
    
    # add the column of split zero for the test dataset into the dataframe
    # just in order to reuse other functions from multiple split case
    setname = "test"
    split_idx = 0
    col_name = f"{setname}_mask_split{split_idx}"
    df_exp[col_name] = 0
    # set all the observed hourly blocks in [start_hour, end_hour] as 1, which belong to the test set
    df_exp.loc[(df_exp["Hour of Day"]>=start_hour) & (df_exp["Hour of Day"]<=end_hour) & (df_exp["step_mask"]==1), col_name] = 1
    
    # just to make it consistent with data_utils.get_hourly_data
    time_dict = {"test": {}} # record the train, valid and test axis for each split
    time_dict["test"][0] = df_exp.loc[df_exp[col_name]==1].index.tolist()

    # correctness check
    assert df_exp.loc[df_exp["step_mask"]==0, f"test_mask_split{split_idx}"].unique() == [0], f"test_mask_split{split_idx} of pid {pid} has invalid step_mask"

    if not conv_feat:
        if return_time_dict:
            # in order to do the stratified sampling for the training data
            return df_exp, step_rate_mean, step_rate_std, time_dict
        else:
            return df_exp, step_rate_mean, step_rate_std


    ####### The following codes are only for ONE-LAYER model #########
    ### Build feature for convolution based models ###
    dataset = {0: {}}
    dataset[0]["test"] = np.array(time_dict["test"][0])
        
    df_conv = df_exp
    # add train, valid and test mask during the computation (Note that all the above masks are for computing the loss and evaluation metrics but not for computing)
    # Train: test are masked out (i.e. in the context window, there could be training and validation context points, the center chunk needs to be masked out in the model)
    # Valid: test are masked out (i.e. in the context window, there could be training and validation context points, the center chunk needs to be masked out in the model)
    # Test: nothing is masked out (i.e. in the context window, there could be training and validation context points, 
    # and also test points which are the center of other test context windows, the center chunk needs to be masked out in the model)
    
    # test
    df_conv[f"test_mask_comp_split{0}"] = 1
    #df_conv.loc[df_conv["time_axis"].isin(dataset[i]["test"]), f"test_mask_comp_split{i}"] = 0
    # set the mask corresponding to the original missing values as 0
    # note that hourly blocks out of [start_hour, end_hour] are also visible
    df_conv.loc[df_conv["step_mask"]==0, f"test_mask_comp_split{0}"] = 0

    # correctness check
    assert (len(df_conv.loc[df_conv["step_mask"]==0]) + len(df_conv.loc[df_conv[f"test_mask_comp_split{0}"]==1]))== len(df_conv), "test comp split mask is not correct!"
    
    # features used to compute the interpolation
    feature_list = []
    feature_list += ["step_rate_norm"]
    feature_list += ["test_mask_comp_split0"]
    # feature_list += ["season", "Day of Week", "Hour of Day", "time_axis"]
    feature_list += ["Day of Week", "Hour of Day", "time_axis"]
    feature_list += ["heart_rate_norm"]
    # masks to compute the loss and evaluation metrics
    feature_list += ["test_mask_split0"]
    # groundtruth
    feature_list += ["step_rate", "valid_minutes", "steps"]

    # get 2D features using pivot table
    df_conv = df_conv[feature_list + ["date", "time"]]
    feat2d_list = []
    for feat in feature_list:
        feat2d_list.append(pd.pivot_table(df_conv[["date", "time", feat]], index=["time"], columns=["date"], dropna=False).values[None, ...])
    # concatenate the pivot 
    conv_feat = np.concatenate(feat2d_list, axis=0)

    # we move the padding feature part into the model
    return conv_feat.astype("float32"), feature_list, step_rate_mean, step_rate_std

def get_multiple_pid(if_high_miss=True, start_hour=6, end_hour=22, conv_feat=True, return_time_dict=False):
    """
    get the data for multiple participant id, refer to get_hourly_data for the arguments.
    Args:
        - pid_list: python list containing all the pids to be processed.
        - if_high_miss: bool, if external validation is on participants with high miss rate or low miss rate
    """
    if if_high_miss:
        pull_file(HIGH_MISS_START_END_FILE) # get the start day and end day file
        pull_file(HIGH_MISS_SHIFT_FILE) # get the shift of day of the week file
        df_start_end_day = pd.read_parquet(f"{FILE_CACHE}/{HIGH_MISS_START_END_FILE}")
        df_best_shift_dw = pd.read_parquet(f"{FILE_CACHE}/{HIGH_MISS_SHIFT_FILE}") 
    else:
        pull_file(LOW_MISS_START_END_FILE) # get the start day and end day file
        pull_file(LOW_MISS_SHIFT_FILE) # get the shift of day of the week file
        df_start_end_day = pd.read_parquet(f"{FILE_CACHE}/{LOW_MISS_START_END_FILE}")
        df_best_shift_dw = pd.read_parquet(f"{FILE_CACHE}/{LOW_MISS_SHIFT_FILE}") 
    
    pid_list = df_start_end_day.index.tolist()
    
    pid_data = [get_hourly_data(pid, df_start_end_day, df_best_shift_dw, start_hour, end_hour, conv_feat, return_time_dict) for pid in tqdm(pid_list)]  # (conv_feat, feature_list, step_rate_mean, step_rate_std)
    
    return pid_data

if __name__ == "__main__":
    # for high miss rate
    pid_data_high_miss = get_multiple_pid(if_high_miss=True, start_hour=6, end_hour=22, conv_feat=True, return_time_dict=False)
    print(f"high miss cohort has {len(pid_data_high_miss)} participants")
    with open(f"{FILE_CACHE}/pid_data_extvalid_high_miss.pkl", "wb") as fout:
        pickle.dump(pid_data_high_miss, fout)
    os.system(f"gsutil -m cp {FILE_CACHE}/pid_data_extvalid_high_miss.pkl {os.getenv('WORKSPACE_BUCKET')+'/data/'}")
    
    # for low miss rate
    pid_data_low_miss = get_multiple_pid(if_high_miss=False, start_hour=6, end_hour=22, conv_feat=True, return_time_dict=False)
    print(f"low miss cohort has {len(pid_data_low_miss)} participants")
    with open(f"{FILE_CACHE}/pid_data_extvalid_low_miss.pkl", "wb") as fout:
        pickle.dump(pid_data_low_miss, fout)
    os.system(f"gsutil -m cp {FILE_CACHE}/pid_data_extvalid_low_miss.pkl {os.getenv('WORKSPACE_BUCKET')+'/data/'}")


    
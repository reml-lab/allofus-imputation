import os
import copy
import pickle
from tqdm.notebook import tqdm

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pyarrow.parquet as pq

from utils import data_utils


def get_dist_active_hours(pid_list):
    # Get the active level of hour of the day over all participants
    # we need to define which subset of the day we would like to care about
    # the way to do it is to get the median, and quantiles of step counts for each hour of the day from all the participants

    if not data_utils.pull_file("active_hour_stats.pkl"):
        data_utils.pull_file(data_utils.DATA_FILE)
        parquet_file = pq.ParquetFile(f"{data_utils.FILE_CACHE}/{data_utils.DATA_FILE}")

        # note that participants could have some hours with no valid minutes at all
        active_hours_dict = {hour: [] for hour in range(24)}

        print("Extracting info from data file...")
        for pid in tqdm(pid_list):
            df_part = data_utils.get_data_by_id(pid)
            df_part = df_part.loc[df_part["valid_minutes"] > 0]
            part_median = df_part.groupby("Hour of Day")["steps"].agg("median")
            for hour, median in part_median.items():
                active_hours_dict[hour].append(median)

        # get the median and quantile for each hour 
        hour_median = [np.median(active_hours_dict[hour]) for hour in range(24)]
        hour_upper = [np.quantile(active_hours_dict[hour], 0.95) for hour in range(24)]
        hour_lower = [np.quantile(active_hours_dict[hour], 0.05) for hour in range(24)]

        with open(f"{data_utils.FILE_CACHE}active_hour_stats.pkl", "wb") as fout:
            pickle.dump({"median":hour_median, "95% quantile":hour_upper, "5% quantile":hour_lower}, fout)

        os.system(f"gsutil -m cp {data_utils.FILE_CACHE}/active_hour_stats.pkl {os.getenv('WORKSPACE_BUCKET')+'/data/'}")
    else:
        with open(f"{data_utils.FILE_CACHE}/active_hour_stats.pkl", "rb") as fin:
            hour_dict = pickle.load(fin)
        hour_median = hour_dict["median"]
        hour_upper = hour_dict["95% quantile"]
        hour_lower = hour_dict["5% quantile"]
    
    return hour_median, hour_upper, hour_lower


# Get the distribution of the continuously missing daily blocks
def get_all_cont_zero_lens(seq):
    seq.insert(0, -1)
    seq.append(-1)
    cont_zeros_list = []
    for i in range(len(seq)):
        if seq[i] == 0:
            if seq[i-1]!=0:
                cont_zeros_num = 1
                if seq[i+1]!=0:
                    cont_zeros_list.append(cont_zeros_num)
            else:
                cont_zeros_num += 1
                if seq[i+1]!=0:
                    cont_zeros_list.append(cont_zeros_num)
    return cont_zeros_list


def get_cont_miss_daily_blocks(pid_list):
    if data_utils.pull_file("all_miss_block_lens.pkl") and data_utils.pull_file("mode_miss_block_lens.pkl"):
        with open(f"{data_utils.FILE_CACHE}/all_miss_block_lens.pkl", "rb") as fin:
            all_miss_block_lens = pickle.load(fin)
        with open(f"{data_utils.FILE_CACHE}/mode_miss_block_lens.pkl", "rb") as fin:
            mode_miss_block_lens = pickle.load(fin)
    else:
        data_utils.pull_file(data_utils.DATA_FILE)
        parquet_file = pq.ParquetFile(f"./file_cache/{data_utils.DATA_FILE}")

        all_miss_block_lens = []  # store all the lengths of missing daily blocks 
        mode_miss_block_lens = []  # store the mode of all the lengths of missing daily blocks
        print("Extracting info from data file...")
        for pid in tqdm(pid_list):
            df_part = data_utils.get_data_by_id(pid)
            # we only care about 6:00 to 22:00
            df_part = df_part.loc[(df_part["Hour of Day"]>=6) & (df_part["Hour of Day"]<=22)]
            # get daily valid minutes 
            daily_valid_minutes = df_part.groupby("Study day")["valid_minutes"].agg("sum").values
            valid_day_indicators = (daily_valid_minutes > 0).astype(int).tolist()
            cont_zeros_list = get_all_cont_zero_lens(valid_day_indicators)

            all_miss_block_lens.extend(cont_zeros_list)
            # some participants have non zero daily valid minutes
            if len(cont_zeros_list)!=0:
                mode_miss_block_lens.append(stats.mode(cont_zeros_list)[0].item())

        with open(f"{data_utils.FILE_CACHE}/all_miss_block_lens.pkl", "wb") as fout:
            pickle.dump(all_miss_block_lens, fout)
        with open(f"{data_utils.FILE_CACHE}/mode_miss_block_lens.pkl", "wb") as fout:
            pickle.dump(mode_miss_block_lens, fout)

        os.system(f"gsutil -m cp {data_utils.FILE_CACHE}/all_miss_block_lens.pkl {os.getenv('WORKSPACE_BUCKET')+'/data/'}")
        os.system(f"gsutil -m cp {data_utils.FILE_CACHE}/mode_miss_block_lens.pkl {os.getenv('WORKSPACE_BUCKET')+'/data/'}")
        
    
    return all_miss_block_lens, mode_miss_block_lens


def get_longest_valid_days(day_binarys, val_days_lower_bound=180, tol=3):
    """
    Given a binary sequence which indicates if a day has more than 1 valid minute, determine the longest valid
    days which has longer than 6 months and can have at most consecutive 3 missing days, also has less than 10% 
    missing rate
    Args:
        - day_binarys: python list, each element indicates if the study day has at least one valid minute
        - val_days_lower_bound: the lower bound of the length of longest valid days
        - tol: how many consecutive missing days we can have at most
    """
    # insert a special token indicating the start 
    day_binarys.insert(0, -1)
    
    # define some intermediate variables
    last_start_day = -1
    last_end_day = -1
    max_len = 0

    start_day = -1
    end_day = -1
    num_zero = -1 # compute the number of accumulated zeros, when it equals to -1, it means before the next valid sequence

    for i in range(len(day_binarys)):
        if day_binarys[i] == 1:
            if day_binarys[i-1] != 1:
                if num_zero == -1:
                    start_day = i

            end_day = i
            num_zero = 0
            if max_len < end_day - start_day + 1:
                last_start_day = start_day
                last_end_day = end_day
                max_len =  end_day - start_day + 1

        elif day_binarys[i]==0:
            # since it is possible day_binarys[i] == -1
            if day_binarys[i-1] == 1:
                num_zero = 1
            elif day_binarys[i-1]==0:
                if num_zero != -1:
                    num_zero += 1
                    if num_zero > tol:
                        if max_len < end_day - start_day + 1:
                            last_start_day = start_day
                            last_end_day = end_day
                            max_len = end_day - start_day + 1
                        num_zero = -1
    
    if max_len >= val_days_lower_bound:
        # Note we need to shift by 1 day due to the start token
        return last_start_day-1, last_end_day-1, max_len
    else:
        # indicate there is no valid day sequence
        return -1, -1, -1
    

def get_valid_study_days(pid_list, val_days_lower_bound=180, tol=3, daily_minutes_lower_bound=240, miss_rate_upper_bound=0.2, from_scratch=False):
    """
    Args:
        - val_days_lower_bound: the lower bound of the length of longest valid days (default: 180 days)
        - tol: how many consecutive missing days we can have at most (default: 3 days)
        - daily_minutes_lower_bound: the lower bound of the daily wearing minutes (default: 4 hours)
        - miss_rate_upper_bound: the upper bound of the missing rate in the final data (default: 0.2)
        - from_scratch: whether to rebuild the dataframe (default: False)
    """
    
    if not data_utils.pull_file("valid_start_end_day.parquet") or from_scratch:
    
        valid_pid_list = []
        start_day_list = []
        end_day_list = []
        max_len_list = []
        miss_rate_list = []

        data_utils.pull_file(data_utils.DATA_FILE)
        parquet_file = pq.ParquetFile(f"./file_cache/{data_utils.DATA_FILE}")

        for pid in tqdm(pid_list):
            df_part = data_utils.get_data_by_id(pid)
            # we only care about 6:00 to 22:00
            df_part = df_part.loc[(df_part["Hour of Day"]>=6) & (df_part["Hour of Day"]<=22)]
            # get the valid day indicator sequence
            daily_valid_minutes = df_part.groupby("Study day")["valid_minutes"].agg("sum").values
            valid_day_indicators = (daily_valid_minutes>=daily_minutes_lower_bound).astype(int).tolist()
            # compute the longest valid day sequence
            start_day, end_day, max_len = get_longest_valid_days(valid_day_indicators, val_days_lower_bound, tol)
            # get the valid participant
            if max_len != -1:
                # we need to compute the missing rate
                df_part = df_part.loc[(df_part["Study day"]>=start_day) & (df_part["Study day"]<=end_day)]
                miss_rate = len(df_part.loc[df_part["valid_minutes"]==0]) / len(df_part)
                if miss_rate <= miss_rate_upper_bound:
                    valid_pid_list.append(pid)
                    start_day_list.append(start_day)
                    end_day_list.append(end_day)
                    max_len_list.append(max_len)
                    miss_rate_list.append(miss_rate)

            # print(f"pid:{pid} | start_day:{start_day} | end_day:{end_day} | max_len:{max_len} | miss_rate:{miss_rate}")

        df = pd.DataFrame({"Participant ID":valid_pid_list,
                           "Start Day":start_day_list,
                           "End Day": end_day_list,
                           "Max Len": max_len_list,
                           "Miss Rate": miss_rate_list})
        df = df.set_index("Participant ID")

        df.to_parquet(os.path.join(data_utils.FILE_CACHE, "valid_start_end_day.parquet"))
        os.system(f"gsutil -m cp {data_utils.FILE_CACHE}/valid_start_end_day.parquet {os.getenv('WORKSPACE_BUCKET')+'/data/'}")
    
    else:
        df = pd.read_parquet(f"{data_utils.FILE_CACHE}/valid_start_end_day.parquet")
        
    return df
    

def align_dayweek(ref, seq):
    """
    Align the sequence with the reference
    Args:
        - ref: the reference sequence
        - seq: sequence to be aligned
    """
    best_dist = np.inf
    # we need to apply the max-min norm to both sequence for the correct shift
    ref_ds = (ref - np.min(ref))/ (np.max(ref) - np.min(ref))  # downshift version
    seq_ds = (seq - np.min(seq)) / (np.max(seq) - np.min(seq))  # downshift version
    for i in range(7):
        seq_shift = np.roll(seq_ds, i)
        mae_dist = np.abs(seq_shift - ref_ds).sum()
        if mae_dist < best_dist:
            best_shift = i
            best_dist = mae_dist
        # print(i, mae_dist)
    return best_shift


def get_step_rate_dayweek(pid, df_start_end_day):
    # read the dataframe for the participant
    df_part = data_utils.get_data_by_id(pid)

    # limit the data to be between 6:00 to 22:00 and between start_day and end_day
    df_part = df_part.loc[(df_part["Hour of Day"]>=6) & (df_part["Hour of Day"]<=22)]
    df_part = df_part.loc[(df_part["Study day"]>=df_start_end_day.loc[pid, "Start Day"])&(df_part["Study day"]<=df_start_end_day.loc[pid, "End Day"])]

    # compute the step count per valid minute
    sr_dw_seq = df_part.groupby("Day of Week").apply(lambda x: x["steps"].sum() / x["valid_minutes"].sum()).values
    
    return sr_dw_seq


def get_best_shift_dayweek(pid_list, df_start_end_day, ref_pid, from_scratch=False):
    """
    distribution of step rate w.r.t each day of week from pid=1000471 is used for the reference sequence
    Args:
        - ref_pid: participant id whose day of week distribution is used as reference
    """
    
    if not data_utils.pull_file("dayweek_shift.parquet") or from_scratch:
        # get the reference distribution
        ref_seq = get_step_rate_dayweek(ref_pid, df_start_end_day)
        best_shift_list = []
        for pid in tqdm(pid_list):
            # compute the step rate for each day of week
            sr_dw_seq = get_step_rate_dayweek(pid, df_start_end_day)
            # compute the best shift for alignment between reference sequence and the current sequence
            best_shift = align_dayweek(ref_seq, sr_dw_seq)
            best_shift_list.append(best_shift)
        
        df = pd.DataFrame({"Participant ID": pid_list,
                           "Best Shift": best_shift_list})
        df = df.set_index("Participant ID")

        df.to_parquet(os.path.join(data_utils.FILE_CACHE, "dayweek_shift.parquet"))
        os.system(f"gsutil -m cp {data_utils.FILE_CACHE}/dayweek_shift.parquet {os.getenv('WORKSPACE_BUCKET')+'/data/'}")
    
    else:
        df = pd.read_parquet(f"{data_utils.FILE_CACHE}/dayweek_shift.parquet")
        
    return df


def get_total_valid_minutes_part(data_start_end_day):
    """
    we add the total valid minutes to data_start_end_day
    """
    if not data_utils.pull_file("all_stats_qual_part.parquet"):
        df_stats = data_start_end_day.copy()
        df_stats["total_valid_minutes"] = 0
        for pid in tqdm(df_stats.index.to_list()):
            # read the dataframe of the participant
            df_exp = data_utils.get_data_by_id(pid)
            # select the qualified study day
            start_day, end_day = df_stats.loc[pid, ["Start Day", "End Day"]]
            df_exp = df_exp.loc[(df_exp["Study day"]>=start_day)&(df_exp["Study day"]<=end_day)]
            # select the time from 6:00 to 22:00
            df_exp = df_exp.loc[(df_exp["Hour of Day"]>=6)&(df_exp["Hour of Day"]<=22)]
            # get the total valid minutes
            df_stats.loc[pid, "total_valid_minutes"] = df_exp["valid_minutes"].sum()
        # add the number of valid hourly blocks
        # note we need (1- miss_rate)
        df_stats["Valid_Hourly_Blocks"] = df_stats["Max Len"] * 17 * (1-df_stats["Miss Rate"])
        # sort the valid hourly blocks
        df_stats = df_stats.sort_values(by=["Valid_Hourly_Blocks", "total_valid_minutes"], ascending=[False, False])
        # write into the file
        df_stats.to_parquet(os.path.join(data_utils.FILE_CACHE, "all_stats_qual_part.parquet"))
        os.system(f"gsutil -m cp {data_utils.FILE_CACHE}/all_stats_qual_part.parquet {os.getenv('WORKSPACE_BUCKET')+'/data/'}")
    else:
        df_stats = pd.read_parquet(f"{data_utils.FILE_CACHE}/all_stats_qual_part.parquet")
        
    return df_stats


def create_train_cohort():
    # get information of all the extracted participants
    df_info = data_utils.get_data_info()
    
    # select qualified participants
    df_info_qual = df_info.loc[df_info["Num Hours"]>=6000]

    # get the longest period block for each participant which has 
    # (1) longer than 6 months, and can have at most consecutive 3 days missing 
    # (2) less than 20% missing rate
    df_start_end_day = get_valid_study_days(df_info_qual.index.tolist(), from_scratch=True)
    print("total number of cropped sequence is", len(df_start_end_day))
    
    # get the best shift of day of the week for each participant
    # w.r.t the referencened participant (pid=1000471)
    df_best_shift_dayweek = get_best_shift_dayweek(df_start_end_day.index.tolist(), 
                                                   df_start_end_day, 
                                                   ref_pid=1000471, 
                                                   from_scratch=True)
    
    # get the total valid minutes and number of valid hourly blocks
    # hourly blocks out of 6:00 am to 10:00 pm are not counted
    df_start_end_day = get_total_valid_minutes_part(df_start_end_day)
    # sort the valid hourly blocks
    df_start_end_day = df_start_end_day.sort_values(by=["Valid_Hourly_Blocks", "total_valid_minutes"], ascending=[False, False])
    
    # get first 100 participants
    df_cohort = df_start_end_day.head(100)
    # print total valid hours
    print(f'total number of valid hourly blocks {df_cohort["Valid_Hourly_Blocks"].sum()}')
    print(f'total valid hours {df_cohort["total_valid_minutes"].sum()/60:.2f}')
    # write into the file
    df_cohort.to_parquet(os.path.join(data_utils.FILE_CACHE, "df_cohort_top100.parquet"))
    os.system(f"gsutil -m cp {data_utils.FILE_CACHE}/df_cohort_top100.parquet {os.getenv('WORKSPACE_BUCKET')+'/data/'}")
    
    # preprocess the data for the training cohort
    pid_list = df_cohort.index.to_list()
    pid_data = data_utils.get_multiple_pid(pid_list)
    # write it into the google bucket
    with open(f"{data_utils.FILE_CACHE}/pid_data.pkl", "wb") as fout:
        pickle.dump(pid_data, fout)
    # copy it into the google bucket
    os.system(f"gsutil -m cp {data_utils.FILE_CACHE}/pid_data.pkl {os.getenv('WORKSPACE_BUCKET')+'/data/'}")


if __name__ == "__main__":
    # create the training cohort with 100 participants
    # the participants are selected based on the number of valid minutes
    # the cohort is after preprocessing
    create_train_cohort()

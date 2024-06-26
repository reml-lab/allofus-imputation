import os
import copy
from tqdm.notebook import tqdm
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
from statsmodels.tsa.stattools import acf
import pyarrow.parquet as pq

import sys
sys.path.append("../..")

from utils import package_utils, data_utils
from create_train_cohort import get_longest_valid_days, get_total_valid_minutes_part, get_step_rate_dayweek, align_dayweek

def get_valid_study_days(pid_list, val_days_lower_bound=180, tol=3, daily_minutes_lower_bound=240, miss_rate_lower_bound=0.2, from_scratch=False):
    """
    Args:
        - val_days_lower_bound: the lower bound of the length of longest valid days (default: 180 days)
        - tol: how many consecutive missing days we can have at most (default: 3 days)
        - daily_minutes_lower_bound: the lower bound of the daily wearing minutes (default: 4 hours)
        - miss_rate_upper_bound: the upper bound of the missing rate in the final data (default: 0.2)
        - from_scratch: whether to rebuild the dataframe (default: False)
    """
    
    if not data_utils.pull_file("extvalid_valid_start_end_day.parquet") or from_scratch:
    
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
                if miss_rate > miss_rate_lower_bound:
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

        df.to_parquet(os.path.join(data_utils.FILE_CACHE, "extvalid_valid_start_end_day.parquet"))
        #os.system(f"gsutil -m cp {data_utils.FILE_CACHE}/extvalid_valid_start_end_day.parquet {os.getenv('WORKSPACE_BUCKET')+'/data/'}")
    
    else:
        df = pd.read_parquet(f"{data_utils.FILE_CACHE}/extvalid_valid_start_end_day.parquet")
        
    return df


def select_ppl_extvalid_low_missrate(df_stats_ori):
    """
    df_stats: sorted version as above
    """
    df_stats = df_stats_ori.copy()
    # set up the random seed
    np.random.seed(0)

    # we just randomly select 100 participants for the low missing rate cohort
    part_id_list = np.random.choice(df_stats.index.tolist(), size=100, replace=False).tolist()
    
    return df_stats.loc[part_id_list].sort_values(by=["total_valid_minutes"], ascending=[False])


# define a function to randomly select ppl
def random_select_pid(df_stats_interval, size):
    np.random.seed(0)
    pid_candid = df_stats_interval.index.tolist()
    return np.random.choice(pid_candid, size, replace=False).tolist()


def get_best_shift_dayweek(pid_list, df_start_end_day_ref, df_start_end_day_extvalid, ref_pid, if_high_miss=True, from_scratch=False):
    """
    distribution of step rate w.r.t each day of week from pid=1000471 is used for the reference sequence
    Args:
        - df_start_end_day_ref: for the original generated 7316 ppl for selecting 100 ppl training cohort
        - df_start_end_day_extvalid: for the external validation, either high missing rate or low missing rate
        - ref_pid: participant id whose day of week distribution is used as reference
    """
    
    if if_high_miss:
        filename = "dayweek_shift_extvalid_high_missrate.parquet"
    else:
        filename = "dayweek_shift_extvalid_low_missrate.parquet"
    
    if not data_utils.pull_file(filename) or from_scratch:
        # get the reference distribution
        ref_seq = get_step_rate_dayweek(ref_pid, df_start_end_day_ref)
        best_shift_list = []
        for pid in tqdm(pid_list):
            # compute the step rate for each day of week
            sr_dw_seq = get_step_rate_dayweek(pid, df_start_end_day_extvalid)
            # compute the best shift for alignment between reference sequence and the current sequence
            best_shift = align_dayweek(ref_seq, sr_dw_seq)
            best_shift_list.append(best_shift)
        
        df = pd.DataFrame({"Participant ID": pid_list,
                           "Best Shift": best_shift_list})
        df = df.set_index("Participant ID")

        df.to_parquet(os.path.join(data_utils.FILE_CACHE, filename))
        os.system(f"gsutil -m cp {data_utils.FILE_CACHE}/{filename} {os.getenv('WORKSPACE_BUCKET')+'/data/'}")
    
    else:
        df = pd.read_parquet(f"{data_utils.FILE_CACHE}/{filename}")
        
    return df


def create_extvalid_test_cohort():
    ids = data_utils.get_participant_ids()
    print(f"Got {len(ids)} particiant ids")

    df_info = data_utils.get_data_info()
    # qualified participants are those whose number of total hours 
    # should be no smaller than 71*24 = 1704 hrs
    df_info_qual = df_info.loc[df_info["Num Hours"]>=1704]

    # get the start and the end day for each qualified participant
    df_start_end_day = get_valid_study_days(df_info_qual.index.tolist(), 
                                            val_days_lower_bound=71, 
                                            tol=40, 
                                            daily_minutes_lower_bound=60,
                                            from_scratch=True)
    
    for miss_rate in [0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
        df_miss_rate = df_start_end_day.loc[(df_start_end_day["Miss Rate"]>=miss_rate) & (df_start_end_day["Miss Rate"]<=(miss_rate+0.1))]
        num_ppl = len(df_miss_rate)
        print(f"Miss rate: {miss_rate:.1f} ~ {(miss_rate + 0.1):.1f} | # of participants: {num_ppl} | Max # of days: {df_miss_rate['Max Len'].max()} | Min # of days: {df_miss_rate['Max Len'].min()}")
    
    # The final external validation set includes: 
    # (1). participants with missing rate from 6:00 am to 10:00 pm lower than 0.2 
    # (to see if the model can generalize to other ppl when the missing rate is similar to the original training data): we randomly select 100 participants from it
    # (2). participants with missing rate from 6:00 am to 10:00 pm larger than 0.2: 
    # we randomly select 100 participants from [0.2, 0.4), [0.4, 0.6), [0.6, 0.8), [0.8, 1.0)
    data_utils.pull_file("all_stats_qual_part.parquet")
    ###### stats from the original qualified participants ######
    df_stats_low_missrate = pd.read_parquet(f"{data_utils.FILE_CACHE}/all_stats_qual_part.parquet")
    # sort the valid hourly blocks
    df_stats_low_missrate = df_stats_low_missrate.sort_values(by=["Valid_Hourly_Blocks", "total_valid_minutes"], ascending=[False, False])
    # stats from the ppl with missing rates higher than 0.2
    df_stats_high_missrate = get_total_valid_minutes_part(df_start_end_day)

    # it takes 20 min to generate, so we push it into the file
    df_stats_high_missrate.to_parquet(os.path.join(data_utils.FILE_CACHE, "df_stats_high_missrate.parquet"))
    os.system(f"gsutil -m cp {data_utils.FILE_CACHE}/df_stats_high_missrate.parquet {os.getenv('WORKSPACE_BUCKET')+'/data/'}")

    # read in the original 100 participant
    data_utils.pull_file("df_cohort_top100.parquet")
    df_cohort = pd.read_parquet(f"{data_utils.FILE_CACHE}/df_cohort_top100.parquet")
    pid_list = df_cohort.index.tolist()

    ## remove the first 100 ppl which are already in the training cohort
    df_stats_low_missrate_extvalid = df_stats_low_missrate[100:]
    df_stats_low_missrate_extvalid = select_ppl_extvalid_low_missrate(df_stats_low_missrate_extvalid)
    # store it as a file and upload to the google bucket
    df_stats_low_missrate_extvalid.to_parquet(os.path.join(data_utils.FILE_CACHE, "df_cohort_extvalid_low_missrate.parquet"))
    os.system(f"gsutil -m cp {data_utils.FILE_CACHE}/df_cohort_extvalid_low_missrate.parquet {os.getenv('WORKSPACE_BUCKET')+'/data/'}")

    # store all the selected pids for the high missing rate during external validation
    pid_high_missrate_list = []
    miss_rate_left_edges = np.arange(0.2, 0.9, 0.1)

    for left_edge in miss_rate_left_edges:
        if left_edge == miss_rate_left_edges[-1]:
            right_edge = 1.0
            num_part = 100
        else:
            right_edge = left_edge + 0.1
            num_part = 50
        # print(left_edge, right_edge)
        # get the dataframe consisting of pid in this miss rate interval
        df_stats_interval = df_stats_high_missrate.loc[(df_stats_high_missrate["Miss Rate"]>=left_edge)&(df_stats_high_missrate["Miss Rate"]<right_edge)]
        # remove the ppl who only have 0.0 steps in the data
        if 1252890 in df_stats_interval.index.tolist():
            df_stats_interval = df_stats_interval.loc[df_stats_interval.index!=1252890]
        # randomly select 50 ppl
        pid_interval = random_select_pid(df_stats_interval, size=num_part)
        pid_high_missrate_list.extend(pid_interval)

    #### get the corresponding df_stats ####
    df_stats_high_missrate_extvalid= df_stats_high_missrate.loc[pid_high_missrate_list]
    for i in [0.2, 0.4, 0.6, 0.8]:
        print(len(df_stats_high_missrate_extvalid.loc[(df_stats_high_missrate_extvalid["Miss Rate"]>=i) & (df_stats_high_missrate_extvalid["Miss Rate"]<(i+0.2))]))
    
    # store it as a file and upload to the google bucket
    df_stats_high_missrate_extvalid.to_parquet(os.path.join(data_utils.FILE_CACHE, "df_cohort_extvalid_high_missrate.parquet"))
    os.system(f"gsutil -m cp {data_utils.FILE_CACHE}/df_cohort_extvalid_high_missrate.parquet {os.getenv('WORKSPACE_BUCKET')+'/data/'}")

    #### we need to pull the original valid_start_end_day ####
    data_utils.pull_file("valid_start_end_day.parquet")
    df_start_end_day_ref = pd.read_parquet(f"{data_utils.FILE_CACHE}/valid_start_end_day.parquet")

    ##### high missing rate #####
    data_utils.pull_file("df_cohort_extvalid_high_missrate.parquet")
    df_cohort_extvalid_high_missrate = pd.read_parquet(f"{data_utils.FILE_CACHE}/df_cohort_extvalid_high_missrate.parquet")

    df_best_shift_dayweek_high_missrate = get_best_shift_dayweek(df_cohort_extvalid_high_missrate.index.tolist(), 
                                                                 df_start_end_day_ref,
                                                                 df_cohort_extvalid_high_missrate, 
                                                                 ref_pid=1000471, 
                                                                 if_high_miss=True,
                                                                 from_scratch=True)
    #### low missing rate ####
    # pull the low miss rate df_start_end_day
    data_utils.pull_file("df_cohort_extvalid_low_missrate.parquet")
    df_cohort_extvalid_low_missrate = pd.read_parquet(f"{data_utils.FILE_CACHE}/df_cohort_extvalid_low_missrate.parquet")
    pid_extvalid_low_missrate = df_cohort_extvalid_low_missrate.index.tolist()
    
    df_dayweek_shift = pd.read_parquet(f"{data_utils.FILE_CACHE}/dayweek_shift.parquet")

    # store it into a file and push it into the google bucket
    df_dayweek_shift_extvalid_low_missrate = df_dayweek_shift.loc[pid_extvalid_low_missrate]
    filename = "dayweek_shift_extvalid_low_missrate.parquet"
    df_dayweek_shift_extvalid_low_missrate.to_parquet(os.path.join(data_utils.FILE_CACHE, filename))
    os.system(f"gsutil -m cp {data_utils.FILE_CACHE}/{filename} {os.getenv('WORKSPACE_BUCKET')+'/data/'}")


if __name__ == "__main__":
    # create the completely held-out test cohort with 500 participants
    create_extvalid_test_cohort()







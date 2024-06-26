"""
This script to create KNN input features 
from normalized step rate for multiple participnats
for the External Validation
"""

import numpy as np
import pandas as pd
import faiss
import argparse
import pickle
import time
from tqdm import tqdm
import copy
import os
import multiprocessing

import sys
sys.path.append("../..")

from external_validation.extvalid_utils import get_hourly_data # Note we need to use get_hourly_data from extvalid_utils!!
from external_validation.extvalid_utils import HIGH_MISS_START_END_FILE, HIGH_MISS_SHIFT_FILE, LOW_MISS_START_END_FILE, LOW_MISS_SHIFT_FILE
from utils.data_utils import pull_file, FILE_CACHE
from utils.train_utils import new_dir

def get_args():
    """
    parser the arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--ctx-len', type=int, default=12)  # the number of hours on one side of the current hourly block
    parser.add_argument('--mask-weight', type=float, default=1.0)  # weight for the missing indicators
    parser.add_argument('--n-job', type=int, default=1)
    args = parser.parse_args()

    return args


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


# get the median normalized step rate for each day of week and hour of day (7 * 24 factors)
def get_dayweek_hour_median(df_part):
    """
    Fill with day of the week and hour of the day median (median of the all valid hourly blocks)
    The median are obtained from the train and valid, not from the test
    Args:
        - df_part: the dataframe of the participant
        - split_idx: split index
    """
    df_exp = df_part.copy(deep=True)
    # Note that we need to set all the step_mask before 6:00 and after 22:00 as 1 here
    # since for padding the step rate values, we need to use these step rates
    df_exp.loc[((df_exp["Hour of Day"]<6)|(df_exp["Hour of Day"]>22)) & (df_exp["valid_minutes"]>0), "step_mask"] = 1

    # split the dataframe into train, valid and test
    # Note that train, valid and test accounts for all the valid hourly blocks between 6:00 ~ 22:00
    # Note here df_train_valid also includes these blocks before 6:00 and after 22:00
    # df_train_valid = df_exp.loc[(df_exp[f"test_mask_split{split_idx}"]==0)]

    ### Modified: 12/05/2023 ###
    # we actually use all the statistics from the test since there is no train or valid 
    # for external validation
    df_train_valid = df_exp.loc[df_exp[f"test_mask_split{0}"]==1]

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


def get_feat_all_hourly_blocks(df_exp, dayweek_hourly_median, ctx_len=72):
    """
    Get all the necessary features for the KNN algorithm. The features include: normalized step rate, compute
    mask, true mask, time axis, steps, valid minutes
    Args: 
        - df_exp: dataframe of a participant
        - split_idx: split index
        - dayweek_hourly_median: the dictionary recording the median for each dayweek and hour
        - ctx_len: context window length on one side of the current hour (24 means 24 hours before and after the 
                   current hourly block)
    """
    
    df_sr_hd = df_exp.copy(deep=True)

    # fill in the potential filling values for every hourly block
    df_sr_hd["fill_value"] = df_sr_hd.apply(lambda x: dayweek_hourly_median[x["Day of Week"]][x["Hour of Day"]], axis=1)
        
    ext_feat_list = ["step_rate_norm",                     # 0
                     "test_mask_comp_split0",              # 1
                     "Day of Week",                        # 2
                     "Hour of Day",                        # 3
                     "time_axis",                          # 4
                     "heart_rate_norm",                    # 5
                     "test_mask_split0",                   # 6
                     "step_rate",                          # 7
                     "valid_minutes",                      # 8
                     "steps"                               # 9
                    ]  # list for extracted feature

    feat_list = []  # list to store all the features
    fill_value_list = []  # list to store all the possible filling values (if some hourly block is missing, then we fill it with the median of that dw + hd)
    
    # Here, get the feature in the column format (column after column)
    # instead of the feature in the row format (row after row)

    # for time_axis in tqdm(df_sr_hd["time_axis"].tolist()):
    for hour in range(df_sr_hd["Hour of Day"].nunique()):
        for study_day in range(df_sr_hd["Study day"].nunique()):
            
            time_axis = df_sr_hd.loc[(df_sr_hd["Hour of Day"]==hour) & (df_sr_hd["Study day"]==study_day), "time_axis"].item()
            
            # we remove the originally missing data here
            if df_sr_hd.loc[df_sr_hd["time_axis"]==time_axis, "valid_minutes"].item() == 0:
                continue
            
            # if not, we get the correct context window and get the corresponding features
            if time_axis - ctx_len < df_sr_hd["time_axis"].min():
                df_time = df_sr_hd.loc[(df_sr_hd["time_axis"]>=df_sr_hd["time_axis"].min()) & (df_sr_hd["time_axis"]<=time_axis+ctx_len)]
                feat_hd = df_time[ext_feat_list].to_numpy()
                # pad zeros on the left 
                pad_len = 2*ctx_len+1-feat_hd.shape[0]
                feat_hd = np.concatenate([np.zeros((pad_len, feat_hd.shape[1])), feat_hd], axis=0)
                # pad the fill_value on the left
                fill_value = df_time["fill_value"].to_numpy()
                dw_curr = df_time.iloc[0]["Day of Week"]
                hd_curr = df_time.iloc[0]["Hour of Day"]
                fill_value_pad = pad_fill_values_dayweek_hour(dw_curr, hd_curr, pad_len, dayweek_hourly_median, direct="backward")
                fill_value = np.concatenate([fill_value_pad, fill_value])


            elif time_axis + ctx_len > df_sr_hd["time_axis"].max():
                df_time = df_sr_hd.loc[(df_sr_hd["time_axis"]>=time_axis-ctx_len) & (df_sr_hd["time_axis"]<=df_sr_hd["time_axis"].max())]
                feat_hd = df_time[ext_feat_list].to_numpy()
                # pad zeros on the right
                pad_len = 2*ctx_len+1-feat_hd.shape[0]
                feat_hd = np.concatenate([feat_hd, np.zeros((pad_len, feat_hd.shape[1]))], axis=0)
                # pad the fill_value on the right
                fill_value = df_time["fill_value"].to_numpy()
                dw_curr = df_time.iloc[-1]["Day of Week"]
                hd_curr = df_time.iloc[-1]["Hour of Day"]
                fill_value_pad = pad_fill_values_dayweek_hour(dw_curr, hd_curr, pad_len, dayweek_hourly_median, direct="forward")
                fill_value = np.concatenate([fill_value, fill_value_pad])

            else:
                df_time = df_sr_hd.loc[(df_sr_hd["time_axis"]>=time_axis-ctx_len) & (df_sr_hd["time_axis"]<=time_axis+ctx_len)]
                feat_hd = df_time[ext_feat_list].to_numpy()
                fill_value = df_time["fill_value"].to_numpy()
            
            assert feat_hd.shape[0]==2*ctx_len+1, f"split 0 | time axis {time_axis} | wrong feat_hd shape!"
            assert fill_value.shape[0]==2*ctx_len+1, f"split 0 | time axis {time_axis} | wrong fill_value shape"
            
            feat_list.append(feat_hd[None, ...])  # [1, 2*ctx_len+1, 9]  
            fill_value_list.append(fill_value[None, ...]) # [1, 2*ctx_len+1]
            
    hd_feat_mtx = np.concatenate(feat_list, axis=0)
    fill_value_mtx = np.concatenate(fill_value_list, axis=0)
    
    assert np.isnan(hd_feat_mtx).any()==False, f"split 0 | hd_feat_mtx has nans"   
    assert np.isnan(fill_value_mtx).any()==False, f"split 0 | fill_value_mtx has nans"
        
    return hd_feat_mtx.astype("float32"), fill_value_mtx.astype("float32")
        
# the following is based on  
# https://stackoverflow.com/questions/29835423/relocate-zeros-to-the-end-of-the-last-dimension-in-multidimensional-numpy-array?rq=3
def get_knn_qual(nn_mtx, k):
    """
    get k qualified nearest neighbors for each hourly block in nn_mtx
    """
    # step 1: move all -1's to the end of each row 
    mask = ~np.sort(nn_mtx==-1, axis=1)
    out = -1 * np.ones_like(nn_mtx)
    out[mask] = nn_mtx[nn_mtx!=-1]
    # step 2: select the first k elements from each row
    return out[:, :k]


def knn_fit_query(pid, df_start_end_day, df_best_shift_dw, ctx_len, n_neighbors):
    print(f"pid {pid} | split 0 begins ...")
    # write the statistics into the file
    file_obj = open(f"{FILE_CACHE}/{pid}_0_{ctx_len}_{n_neighbors}_multippl_records_extvalid.txt", "w")
    overall_start_time = time.time()
    # set the random seed
    np.random.seed(0)

    # read in dataframe of the participant
    df_exp, step_rate_mean, step_rate_std, time_dict = get_hourly_data( pid, df_start_end_day, df_best_shift_dw, start_hour=6, end_hour=22, conv_feat=False, return_time_dict=True)
    
    # add missing indicator (computational mask) for every split
    df_exp = preprocess_data(df_exp, time_dict)

    # get the dictionary of fill values
    dw_hd_med_dict = get_dayweek_hour_median(df_exp)

    # create the features for each study day on each split for the KNN input
    hd_feat_mtx_split, fill_value_mtx_split = get_feat_all_hourly_blocks(df_exp, dw_hd_med_dict, ctx_len)
    
    print(f"pid {pid} | split 0 gets all hourly blocks")

    # find the nearest neighbors for test
    ### test ###
    # step 1: create the features for the KNN to fit
    test_nsr = hd_feat_mtx_split[:, :, 7] # raw step rate
    test_comp_mask = copy.deepcopy(hd_feat_mtx_split[:, :, 1]) # compute mask
    # we need to use the fill_value for the center hourly block
    test_comp_mask[:, test_comp_mask.shape[1]//2] = 0
    test_nsr_masked = test_nsr * test_comp_mask + fill_value_mtx_split * (1 - test_comp_mask)  # we use dw+hd median to fill in the missing normalized step rate
    # test_knn_input_feat = np.concatenate([test_nsr_masked, mask_weight * test_comp_mask], axis=1) # [59922, 290]
    test_knn_input_feat = test_nsr_masked
    # step 2: remove the normalized step rate and the computation mask of the center block
    # remove_indeces = [int(test_knn_input_feat.shape[1]//4), int(test_knn_input_feat.shape[1]//4+test_knn_input_feat.shape[1]/2)]
    # remove_indeces = [int(test_knn_input_feat.shape[1]//2)]
    # test_knn_input_feat = np.delete(test_knn_input_feat, remove_indeces, axis=1)
    # step 3: fit KNN on the input features
    # neigh = NearestNeighbors(n_neighbors=n_neighbors, algorithm="kd_tree")
    d = test_knn_input_feat.shape[1]
    index = faiss.IndexFlatL2(d)
    start_time = time.time()
    index.add(test_knn_input_feat)  # add vectors to the index
    # print(f"total number of added indeces: {index.ntotal}")
    # neigh.fit(test_knn_input_feat)
    # print(f"split {split_idx} | time to fit the KNN model for {n_neighbors} neighbors: {(time.time()-start_time)} seconds")
    file_obj.write(f"split 0 | time to fit the KNN model for {n_neighbors} neighbors: {(time.time()-start_time)} seconds\n")
    # step 3: search for k nearest neighbors for each queries
    start_time = time.time()
    dist_test, nn_test = index.search(test_knn_input_feat, n_neighbors)  # [59922, 200]
    # dist_test, nn_test = neigh.kneighbors(test_knn_input_feat, return_distance=True)  # [59922, 200]
    # print(f"split {split_idx} | time to search for {n_neighbors} nearest neighbors for all the hourly blocks is {time.time()-start_time} seconds")
    file_obj.write(f"split 0 | time to search for {n_neighbors} nearest neighbors for all the hourly blocks is {time.time()-start_time} seconds\n")
    file_obj.write("\n")

    print(f"pid {pid} | split 0 finishes KNN search")

    # remove unqualified neighbors
    ### Solve the problem of the duplicated hd_encoding rows ###
    # Note that if hd_encode of two different hourly blocks are the same, 
    # then sklearn knn will not differentiate them, which means the nearest neighbor index 
    # of two rows are exactly the same, so we cannot assume the first element is the query index anymore.  
    ### test ###
    # we need to filter out the distance matrix first, since nn_train, nn_test will change after we filter out them
    correct_index = np.arange(nn_test.shape[0])
    dist_test[nn_test == correct_index[..., None]] = -1
    dist_test = get_knn_qual(dist_test, dist_test.shape[-1])
    dist_test = np.concatenate([np.zeros((dist_test.shape[0], 1)), dist_test], axis=1)
    # set the same indeces as original correct index as -1 (i.e. unqualified nearest neighbors) 
    # and remove them
    correct_index = np.arange(nn_test.shape[0])
    nn_test[nn_test == correct_index[..., None]] = -1
    nn_test = get_knn_qual(nn_test, nn_test.shape[-1]) # move all the -1 to the end of each row
    nn_test = np.concatenate([correct_index[..., None], nn_test], axis=1)
    
    # get the mask and the position for each dataset
    # test mask for the split
    test_mask_flatten = (hd_feat_mtx_split[:, hd_feat_mtx_split.shape[1]//2, 6] == 1)
    # get the qualified nearest neighbors and also get the corresponding distance
    # for the test
    dist_test_hourly = dist_test[test_mask_flatten, :]
    nn_test_hourly = nn_test[test_mask_flatten, :]  # note this uses nn_test
    # print the qualified nearest neighbor number 
    #nn_all_hourly_qual = np.concatenate([nn_train_hourly, nn_valid_hourly, nn_test_hourly], axis=0) # [42284, 101]
    nn_all_hourly_qual = nn_test_hourly
    all_nn_num = (nn_all_hourly_qual != -1).astype(int)[:, 1:].sum(axis=-1)
    # print("Qualified Nearest Number of All Qualified Hourly Blocks")
    # print('-' * 50)
    # print(f"split {split_idx} | min: {np.min(all_nn_num)} | max: {np.max(all_nn_num)}")
    # print(f"split {split_idx} | mean: {np.mean(all_nn_num):.2f} | std: {np.std(all_nn_num):.2f}")
    # print(f"split {split_idx} | median: {np.median(all_nn_num)} | 25%: {np.percentile(all_nn_num, 25)} | 75%: {np.percentile(all_nn_num, 75)}")
    # print(f"split {split_idx} | 5%: {np.percentile(all_nn_num, 5)} | 95%: {np.percentile(all_nn_num, 95)}")
    
    file_obj.write("Qualified Nearest Number of All Qualified Hourly Blocks\n")
    file_obj.write('-' * 50 + '\n')
    file_obj.write(f"split 0 | min: {np.min(all_nn_num)} | max: {np.max(all_nn_num)}\n")
    file_obj.write(f"split 0 | mean: {np.mean(all_nn_num):.2f} | std: {np.std(all_nn_num):.2f}\n")
    file_obj.write(f"split 0 | median: {np.median(all_nn_num)} | 25%: {np.percentile(all_nn_num, 25)} | 75%: {np.percentile(all_nn_num, 75)}\n")
    file_obj.write(f"split 0 | 5%: {np.percentile(all_nn_num, 5)} | 95%: {np.percentile(all_nn_num, 95)}\n")
    file_obj.write("\n")
    
    # concatenate the distance matrices of train, valid and test
    #dist_all_hourly_qual = np.concatenate([dist_train_hourly, dist_valid_hourly, dist_test_hourly], axis=0) # [42284, 101]
    dist_all_hourly_qual = dist_test_hourly # [42284, 101]

    # correctness check
    # whether the positions of -1 in nn_all_hourly_qual is the same as dist_all_hourly_qual
    assert (np.where(dist_all_hourly_qual==-1)[0] == np.where(nn_all_hourly_qual==-1)[0]).all(), f"split 0 | -1 position in nn_mtx is different from dist_mtx!"
    assert (np.where(dist_all_hourly_qual==-1)[1] == np.where(nn_all_hourly_qual==-1)[1]).all(), f"split 0 | -1 position in nn_mtx is different from dist_mtx!"
    # get the first num_nn qualified nearest neighbors
    # here, we get the minimum number of qualified neighbors
    # for the generalization
    num_nn = np.min(all_nn_num)
    nn_all_hourly_qual_knn = get_knn_qual(nn_all_hourly_qual, num_nn)
    dist_all_hourly_qual_knn = get_knn_qual(dist_all_hourly_qual, num_nn)
    # correctness check
    assert (nn_all_hourly_qual_knn!=-1).all(), f"split 0 | nn_all_hourly_qual_knn has -1"
    assert (dist_all_hourly_qual_knn!=-1).all(), f"split 0 | dist_all_hourly_qual_knn has -1"
    
    # create features for each context window
    nn_all_h = nn_all_hourly_qual_knn.shape[0]
    nn_all_w = nn_all_hourly_qual_knn.shape[1]
    nn_all_feat = hd_feat_mtx_split[nn_all_hourly_qual_knn.flatten(), hd_feat_mtx_split.shape[1]//2, :]
    nn_all_feat = nn_all_feat.reshape(nn_all_h, nn_all_w, nn_all_feat.shape[1])

    # print(f"total time for split {split_idx} is {(time.time()-overall_start_time):.2f} seconds")
    file_obj.write(f"total time for split 0 is {(time.time()-overall_start_time):.2f} seconds\n")
    file_obj.close()

    # store all the features and distance matrix
    knn_feat_dict = {"nn_all_feat": nn_all_feat.astype("float32"), "dist_all_hourly": dist_all_hourly_qual_knn.astype("float32"), "nn_all_hourly": nn_all_hourly_qual_knn}

    print(f"pid {pid} | split 0 finishes!")
    
    return knn_feat_dict  #, split_idx

def knn_fit_query_pid_all_splits(pid, if_high_miss, df_start_end_day, df_best_shift_dw, ctx_len, n_neighbors):
    knn_feat_pid_list = []
    knn_feat_dict_split = knn_fit_query(pid, df_start_end_day, df_best_shift_dw, ctx_len, n_neighbors)
    knn_feat_pid_list.append(knn_feat_dict_split)
    if if_high_miss:
        filename = f"{FILE_CACHE}/knn_raw_feat_extvalid_high_miss/knn_raw_feat_{pid}_extvalid.pkl"
    else:
        filename = f"{FILE_CACHE}/knn_raw_feat_extvalid_low_miss/knn_raw_feat_{pid}_extvalid.pkl"
    with open(filename, "wb") as fout:
        pickle.dump(knn_feat_pid_list, fout)
    #os.system(f"gsutil -m cp {filename} {os.getenv('WORKSPACE_BUCKET')+'/data/'}")

if __name__ == "__main__":
    ctx_len = 72
    n_neighbors = 50

    if_high_miss = True

    if if_high_miss:
        pull_file(HIGH_MISS_START_END_FILE) # get the start day and end day file
        pull_file(HIGH_MISS_SHIFT_FILE) # get the shift of day of the week file
        df_start_end_day = pd.read_parquet(f"{FILE_CACHE}/{HIGH_MISS_START_END_FILE}")
        df_best_shift_dw = pd.read_parquet(f"{FILE_CACHE}/{HIGH_MISS_SHIFT_FILE}") 
        # build folder to store the loss history and the best model
        new_dir(f"{FILE_CACHE}/knn_feat_extvalid_high_miss")
        print("we are doing high miss rate!")
    else:
        pull_file(LOW_MISS_START_END_FILE) # get the start day and end day file
        pull_file(LOW_MISS_SHIFT_FILE) # get the shift of day of the week file
        df_start_end_day = pd.read_parquet(f"{FILE_CACHE}/{LOW_MISS_START_END_FILE}")
        df_best_shift_dw = pd.read_parquet(f"{FILE_CACHE}/{LOW_MISS_SHIFT_FILE}") 
        # build folder to store the loss history and the best model
        new_dir(f"{FILE_CACHE}/knn_feat_extvalid_low_miss")
        print("we are doing low miss rate!")
    
    pid_list = df_start_end_day.index.tolist()
    print(f"total number of participants: {len(pid_list)}")

    if_high_miss_list = [if_high_miss] * len(pid_list)
    ctx_len_list = [ctx_len] * len(pid_list)
    n_neighbors_list = [n_neighbors] * len(pid_list)
    df_start_end_day_list = [df_start_end_day] * len(pid_list)
    df_best_shift_dw_list = [df_best_shift_dw] * len(pid_list)

    with multiprocessing.Pool(processes=30) as pool:
        pool.starmap(knn_fit_query_pid_all_splits, list(zip(pid_list, if_high_miss_list, df_start_end_day_list, df_best_shift_dw_list, ctx_len_list, n_neighbors_list)))
    

    
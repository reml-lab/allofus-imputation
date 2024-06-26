"""
This script is to sample from the trained Chained Equation model (linear model with stored weight and bias) to 
do the multiple imputation for inference on the exterval validation set
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader

import os
import pickle
import copy
import math
import re
import argparse
import time
from datetime import datetime
import multiprocessing
from multiprocessing import get_context
import glob
from tqdm import tqdm

import sys
sys.path.append("../..")

from utils.data_utils import FILE_CACHE, pull_file
from utils.train_utils import new_dir, mse_loss, mae_loss, lower_upper_bound_func
from baselines.mice.model import MiniBatchSGDRegressor
from external_validation.mice.dataset import MiceDatasetExtValid
from external_validation.extvalid_utils import HIGH_MISS_START_END_FILE, HIGH_MISS_SHIFT_FILE, LOW_MISS_START_END_FILE, LOW_MISS_SHIFT_FILE
    
import warnings
warnings.filterwarnings("ignore")


def mice_inference(X_test, coeff_list, comp_feat_idx, mice_iter, lower_bound, upper_bound):
    """
    Sample from Chained Equation results. 
    Args:
        - X_test: input feature for valid or test, shape: [N, 238]
        - coeff_list: list of (coef_, intercept_) for each trained linear model from each split
        - comp_feat_idx: which index could be treated as y in Chained Equation
        - mice_iter: how many iteration of chained equation we run during the training
        - lower_bound: lower bound of normalized step rate of each context window, shape: [N, 1]
        - upper_bound: higher bound of normalized step rate of each context window, shape: [N, 1]
    """

    # check the correctness of the input
    assert len(coeff_list)==(len(comp_feat_idx) * mice_iter), "problem with coeff_list length!"
    
    # try to preserve the original matrix
    input_feat = X_test.copy()
    # get the mask for the missing positions
    miss_indicator = np.isnan(input_feat).astype("int")
    # fill the nan with the zeros
    input_feat[np.isnan(input_feat)] = 0.0
    
    for iter_idx in range(mice_iter):
        # target_id is for the column as y, col_idx is to get the index of coefficient
        for col_idx, target_id in enumerate(tqdm(comp_feat_idx)): 
            # get missing indicator
            target_miss_ind = miss_indicator[:, target_id]
            # skip the complete to save time
            if target_miss_ind.sum() == 0:
                continue
            # get X
            feat_id = np.arange(input_feat.shape[1])  
            feat_id = feat_id[feat_id!=target_id] # feat_id is for the columns as X
            X = input_feat[:, feat_id]
            # get y
            y = input_feat[:, target_id]
            # get weight and bias
            coeff_index = iter_idx * len(comp_feat_idx) + col_idx
            weight = coeff_list[coeff_index][0][None, ...]
            bias = coeff_list[coeff_index][1]
            # get the prediction
            pred_y = np.matmul(X, weight.T) + bias
            
            ##### add the noise here based on the residual #####
            # if the feature is all missing (which probably indicates it is the target feature we need to impute
            # for all the instances), we don't do the sampling, since we cannot estimate the standard deviateion
            # from the groundtruth
            if target_miss_ind.sum()!= target_miss_ind.shape[0]:
                square_residual = (pred_y.squeeze(1) - y) ** 2
                # compute the estimated std from the observed places
                num_obs = (1-target_miss_ind).sum()
                est_std = np.sqrt(square_residual[target_miss_ind==0].sum() / num_obs) # estimated std
                # sample from the gaussian distribution
                eps = np.random.normal(loc=0, scale=est_std, size=pred_y.shape)
                # add noise and the pred_y (mean of gaussian)
                pred_y += eps
            
            ##### add the nonlinearity #####
            # which is to limit the value into the range between the max and min value
            lower_mask = pred_y < lower_bound
            pred_y[lower_mask] = lower_bound[lower_mask]
            upper_mask = (pred_y > 1.5 * upper_bound)
            pred_y[upper_mask] = 1.5 * upper_bound[upper_mask]
            
            # fill in the missing values in input_feat
            input_feat[:, target_id][miss_indicator[:, target_id]==1] = pred_y.squeeze(1)[miss_indicator[:, target_id]==1]

    return input_feat


# get the wrap up function
def get_sample_mc_avg(num_samples, input_feat_eval, coeff_list, comp_feat_idx, mice_iter, lower_bound, upper_bound):
    np.random.seed(0)
    # only store the last feature which is what we would like to compute
    avg_pred_y = np.zeros_like(input_feat_eval[:,-1])
    for _ in range(num_samples):
        sample = mice_inference(input_feat_eval, coeff_list, comp_feat_idx, mice_iter, lower_bound, upper_bound)
        avg_pred_y += sample[:, -1] / num_samples
        
    return avg_pred_y  # [N, ]


### MAIN ###
#def main(args, split_idx):
def run_mice_sampling(split_idx, mice_iter, num_samples, seed, if_high_miss, kh=9, kw=71, pad_full_weeks=False, num_parts=-1):
    
    ### MAIN ### 
    overall_start_time = time.time()

    if if_high_miss:
        print(f"split {split_idx} | high miss begins ...")
    else:
        print(f"split {split_idx} | low miss begins ...")

    ## deal with the randomization ##
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # pull the pid_list
    if if_high_miss:
        df_cohort = pd.read_parquet(f"{FILE_CACHE}/{HIGH_MISS_START_END_FILE}")
    else:
        df_cohort = pd.read_parquet(f"{FILE_CACHE}/{LOW_MISS_START_END_FILE}")

    pid_list = df_cohort.index.tolist()

    # get the data for the participant
    if if_high_miss:
        pid_data_filename = "pid_data_extvalid_high_miss_revised.pkl"
    else:
        pid_data_filename = "pid_data_extvalid_low_miss_revised.pkl"
    with open(f"{FILE_CACHE}/{pid_data_filename}", "rb") as fin:
        pid_data = pickle.load(fin)
    
    # the following is for the debugging
    if num_parts != -1:
        pid_data = pid_data[:num_parts]
    
    ks = (kh, kw)

    # load in all the weight and bias from the trained model of this split
    ### get the list of coefficient ###
    coef_list = []
    for coef_filename in sorted(glob.glob(f"{FILE_CACHE}/mice_weight_bias_all_splits/split{split_idx}_*.pkl")):
        with open(coef_filename, "rb") as fin:
            coef_list.append(pickle.load(fin))

    ## test process ##
    print(f"split {split_idx} | begin testing ...")
    
    test_dataset = MiceDatasetExtValid(pid_data, ks, pad_full_weeks)
        
    # used to compute micro mae and mse
    test_total_mae = 0
    test_total_mse = 0
    test_total_num = 0
        
    # used to compute macro mae and mse
    test_pid_mae_list = [[] for _ in range(len(pid_data))]
    test_pid_mse_list = [[] for _ in range(len(pid_data))]
    test_pid_num_list = [[] for _ in range(len(pid_data))]
        
    input_feat, step_gt, max_sr, sr_mean, sr_std, pid_ids = test_dataset.input_feat, test_dataset.step_gt, \
                test_dataset.max_sr, test_dataset.sr_mean, test_dataset.sr_std, test_dataset.pid_ids
    
    ### get the comp_feat_idx ###
    comp_feat_idx = tuple(np.arange((24+7), (test_dataset.input_feat.shape[1]))) # feat index to compute as the target

    min_sr = np.zeros_like(max_sr)
    upper_bound = (max_sr - sr_mean) / sr_std
    lower_bound = (min_sr - sr_mean) / sr_std
    # test
    start_time = time.time()
    output = get_sample_mc_avg(num_samples, input_feat, coef_list, comp_feat_idx, mice_iter, lower_bound, upper_bound)
    print(f"split {split_idx} | time to sample {num_samples} samples (test) = {(time.time() - start_time):.2f} seconds")
    # get the unnormalized step rate
    output = output[..., None] * sr_std + sr_mean # [N, 1]
    output = lower_upper_bound_func(output, min_sr, 1.5*max_sr)
    # get the step counts
    output = output * step_gt[:, 1][..., None]

    # store the predicted results into the file
    pred_bundle = {"pred": output,
                    "gt": step_gt[:,0],
                    "pids": pid_ids}
    with open(f"{OUT_PATH}/pred_results_dict_mice_split{split_idx}.pkl", "wb") as fout:
        pickle.dump(pred_bundle, fout)

    output = torch.from_numpy(output)    
    step_gt = torch.from_numpy(step_gt)    
    # micro mae and mse
    test_mae = mae_loss(output.squeeze(1), step_gt[:,0], mask=None, norm=True)
    test_mse = mse_loss(output.squeeze(1), step_gt[:,0], mask=None, norm=True)
    test_total_mae += test_mae * output.shape[0]
    test_total_mse += test_mse * output.shape[0]
    test_total_num += output.shape[0]
    # compute the micro mae and rmse
    test_micro_mae = (test_total_mae / test_total_num).item()
    test_micro_rmse = np.sqrt(test_total_mse.item() / test_total_num)
    # macro mae and mse
    for pid in range(len(pid_data)):
        pid_mask = (pid_ids==pid).astype("int")
        test_pid_mae_list[pid].append(mae_loss(output.squeeze(1), step_gt[:,0], mask=pid_mask.squeeze(1), norm=True).item()) # the mean 
        test_pid_mse_list[pid].append(mse_loss(output.squeeze(1), step_gt[:,0], mask=pid_mask.squeeze(1), norm=True).item()) # the mean
        test_pid_num_list[pid].append(pid_mask.squeeze(1).sum().item())
    # compute the macro mae and rmse
    test_macro_mae_list = []
    test_macro_rmse_list = []
    for pid in range(len(pid_data)):
        test_macro_mae = np.array(test_pid_mae_list[pid]) * np.array(test_pid_num_list[pid])
        test_macro_mae = test_macro_mae.sum() / np.array(test_pid_num_list[pid]).sum()
        test_macro_mae_list.append(test_macro_mae)
        test_macro_rmse = np.array(test_pid_mse_list[pid]) * np.array(test_pid_num_list[pid])
        test_macro_rmse = test_macro_rmse.sum() / np.array(test_pid_num_list[pid]).sum()
        test_macro_rmse_list.append(np.sqrt(test_macro_rmse))
    test_macro_mae = np.mean(test_macro_mae_list)
    test_macro_rmse = np.mean(test_macro_rmse_list)

    print(f"split {split_idx} is finished!")

    overall_time = time.time() - overall_start_time
            
    return test_micro_mae, test_micro_rmse, test_macro_mae, test_macro_rmse, overall_time


def main(if_high_miss):
    # get arguments
    # args = get_args()

    num_split = 10
    num_samples = 5
    mice_iter = 2
    kh, kw = 9, 71
    seed = 0

    # write the statistics into the file
    # file_obj = open(args.output_file, "w")
    timestamp = datetime.now().strftime('%m%d.%H%M%S')

    file_obj = open(f"./results/{timestamp}_mice_sampling_{mice_iter}_{seed}_{num_samples}.txt", "w")

    test_micro_mae_list = []
    test_macro_mae_list = []
    test_micro_rmse_list = []
    test_macro_rmse_list = []

    overall_time_list = []

    for split_idx in range(num_split):
        test_micro_mae, test_micro_rmse, test_macro_mae, test_macro_rmse, overall_time = run_mice_sampling(split_idx, mice_iter, num_samples, seed, if_high_miss)
        test_micro_mae_list.append(test_micro_mae)
        test_macro_mae_list.append(test_macro_mae)
        test_micro_rmse_list.append(test_micro_rmse)
        test_macro_rmse_list.append(test_macro_rmse)

        overall_time_list.append(overall_time)
    
    for i in range(num_split):
        file_obj.write(f"split {i} | run time: {overall_time_list[i]:.2f} seconds\n")
    
    file_obj.write("\n")

    for i in range(num_split):
        #print(f"split {i} | best_epoch: {best_epoch} | best_val_mae: {best_val_nll:.2f} | test_mae_best_epoch: {hist_dict['mae']['test_loss'][best_epoch]:.2f}")
        file_obj.write(f"split {i} | test_micro_mae: {test_micro_mae_list[i]:.2f}\n")

    # MICRO MAE loss
    test_micro_mae_mean = np.mean(test_micro_mae_list)
    test_micro_mae_std = np.std(test_micro_mae_list)

    file_obj.write('\n')
    file_obj.write('MICRO MAE Statistics\n')
    file_obj.write("-"*50 + "\n")
    file_obj.write(f"test  | mean: {test_micro_mae_mean:.2f} | std: {test_micro_mae_std:.2f}\n")

    # MACRO MAE
    file_obj.write("\n")
    for i in range(num_split):
        file_obj.write(f"split {i} | test_macro_mae: {test_macro_mae_list[i]:.2f}\n")
    file_obj.write("\n")
    file_obj.write(f"MACRO MAE Best Epoch Statistics\n")
    file_obj.write("-" * 50 + "\n")
    file_obj.write(f"test  | mean: {np.mean(test_macro_mae_list):.2f} | std: {np.std(test_macro_mae_list):.2f}\n")
    
    # MICRO RMSE 
    file_obj.write("\n")
    for i in range(num_split):
        file_obj.write(f"split {i} | test_micro_rmse: {test_micro_rmse_list[i]:.2f}\n")
    file_obj.write("\n")
    file_obj.write(f"MICRO RMSE Best Epoch Statistics\n")
    file_obj.write("-" * 50 + "\n")
    file_obj.write(f"test  | mean: {np.mean(test_micro_rmse_list):.2f} | std: {np.std(test_micro_rmse_list):.2f}\n")
    
    # MACRO RMSE
    file_obj.write("\n")
    for i in range(num_split):
        file_obj.write(f"split {i} | test_macro_rmse: {test_macro_rmse_list[i]:.2f}\n")
    file_obj.write("\n")
    file_obj.write(f"MACRO RMSE Best Epoch Statistics\n")
    file_obj.write("-" * 50 + "\n")
    file_obj.write(f"test  | mean: {np.mean(test_macro_rmse_list):.2f} | std: {np.std(test_macro_rmse_list):.2f}\n")
    
    file_obj.close()

if __name__ == "__main__":
    
    if_high_miss = True

    # build folder to store the predicted results
    new_dir(f"./results")
    OUT_PATH = f"./results/mice_pred_results_extvalid"
    if if_high_miss:
        OUT_PATH += "_high_miss_revised"
    else:
        OUT_PATH += "_low_miss_revised"
    new_dir(OUT_PATH)

    # pull the corresponding files
    if if_high_miss:
        pull_file(HIGH_MISS_START_END_FILE)
    else:
        pull_file(LOW_MISS_START_END_FILE)

    # get the data for the participant
    if if_high_miss:
        pid_data_filename = "pid_data_extvalid_high_miss_revised.pkl"
    else:
        pid_data_filename = "pid_data_extvalid_low_miss_revised.pkl"
    pull_file(pid_data_filename)

    # get the mice parameters
    foldername = "mice_weight_bias_all_splits"
    if not os.path.exists(f"{FILE_CACHE}/{foldername}"):
        os.system(f"gsutil -m cp -r {os.getenv('WORKSPACE_BUCKET')}/data/{foldername} {FILE_CACHE}")  

    main(if_high_miss)
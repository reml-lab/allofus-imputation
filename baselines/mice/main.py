"""
This script is to train and evaluation MICE model.
Please use 64 CPUs with 416 GB RAM to run this script.
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

import sys
sys.path.append("../..")

from utils.data_utils import FILE_CACHE, pull_file
from utils.train_utils import new_dir, mse_loss, mae_loss, lower_upper_bound_func
from baselines.mice.model import MiniBatchSGDRegressor
from baselines.mice.dataset import MiceDataset
    
import warnings
warnings.filterwarnings("ignore")


def get_args():
    """
    parser the arguments to tune the models
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--kh', type=int, default=9)
    parser.add_argument('--kw', type=int, default=71)
    parser.add_argument('--pad-full-weeks', action="store_true")
    parser.add_argument('--num-split', type=int, default=10)
    parser.add_argument('--mice-iter', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=10000)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--store-model', action='store_true')
    
    parser.add_argument('--num-parts', type=int, default=-1) # used for debugging

    args = parser.parse_args()

    return args


def run_mice(split_idx, mice_iter, epochs, batch_size, lr, seed, store_model, kh=9, kw=71, pad_full_weeks=False, num_parts=-1):
    
    overall_start_time = time.time()

    print(f"split {split_idx} begins ...")

    # deal with the randomization
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # pull the pid_list
    pull_file("df_cohort_top100.parquet")
    df_cohort = pd.read_parquet(f"{FILE_CACHE}/df_cohort_top100.parquet")
    pid_list = df_cohort.index.tolist()

    # get the data for the participant
    pull_file("pid_data.pkl")
    with open(f"{FILE_CACHE}/pid_data.pkl", "rb") as fin:
        pid_data = pickle.load(fin)
    # the following is for the debugging
    if num_parts != -1:
        pid_data = pid_data[:num_parts]
    ks = (kh, kw)

    ## Define the model ##
    # define the estimator
    mbsgd = MiniBatchSGDRegressor(store_model=store_model,
                                  split_idx=split_idx,
                                  num_epochs=epochs, 
                                  batch_size=batch_size, 
                                  learning_rate="invscaling",
                                  eta0=lr,
                                  loss='epsilon_insensitive', 
                                  epsilon=0.01)  # note there is a L2 regularization in the model by default
    
    # define the iterative imputer
    imp = IterativeImputer(estimator=mbsgd, 
                           max_iter=mice_iter, 
                           verbose=0, 
                           # initial_strategy='median', 
                           initial_strategy='constant',  # by default, it will fill with zeros
                           random_state=0,  # no use here
                           imputation_order="roman",
                           skip_complete=True)  # note we set it as true to avoid computing one hot vectors
   
    print("finish defining the model!")

    ## Train process ##
    train_dataset = MiceDataset(pid_data, split_idx, "train", ks, pad_full_weeks)

    input_feat, step_gt, max_sr, sr_mean, sr_std, pid_ids = train_dataset.input_feat, train_dataset.step_gt, \
                train_dataset.max_sr, train_dataset.sr_mean, train_dataset.sr_std, train_dataset.pid_ids
    
    # hack the sklearn imputator
    # we need to set normalized step rate of the center hourly block of the first training sample as np.nan
    # so that we can set skip_complete as True in IterativeImputer to skip computing hour of the day and 
    # day of the week
    input_feat[0, -1] = np.nan
    # train process
    start_time = time.time()
    imp.fit(input_feat)
    print(f"split {split_idx} | train time: {(time.time() - start_time):.2f} seconds")

    ## Valid process ##
    print(f"split {split_idx} | begin validation ...")

    valid_dataset = MiceDataset(pid_data, split_idx, "valid", ks, pad_full_weeks)
        
    # used to compute micro mae and mse
    valid_total_mae = 0
    valid_total_mse = 0
    valid_total_num = 0
        
    # used to compute macro mae and mse
    valid_pid_mae_list = [[] for _ in range(len(pid_data))]
    valid_pid_mse_list = [[] for _ in range(len(pid_data))]
    valid_pid_num_list = [[] for _ in range(len(pid_data))]
        
    input_feat, step_gt, max_sr, sr_mean, sr_std, pid_ids = valid_dataset.input_feat, valid_dataset.step_gt, \
                valid_dataset.max_sr, valid_dataset.sr_mean, valid_dataset.sr_std, valid_dataset.pid_ids
    
    output = imp.transform(input_feat)
    # get the unnormalized step rate
    output = output[:, -1][..., None] * sr_std + sr_mean # [N, 1]
    output = lower_upper_bound_func(output, 0.0, max_sr)
    # get the step counts
    output = output * step_gt[:, 1][..., None]
    output = torch.from_numpy(output)    
    step_gt = torch.from_numpy(step_gt)    
    # micro mae and mse
    valid_mae = mae_loss(output.squeeze(1), step_gt[:,0], mask=None, norm=True)
    valid_mse = mse_loss(output.squeeze(1), step_gt[:,0], mask=None, norm=True)
    valid_total_mae += valid_mae * output.shape[0]
    valid_total_mse += valid_mse * output.shape[0]
    valid_total_num += output.shape[0]
    # compute the micro mae and rmse
    valid_micro_mae = (valid_total_mae / valid_total_num).item()
    valid_micro_rmse = np.sqrt(valid_total_mse.item() / valid_total_num)
    # macro mae and mse
    for pid in range(len(pid_data)):
        pid_mask = (pid_ids==pid).astype("int")
        valid_pid_mae_list[pid].append(mae_loss(output.squeeze(1), step_gt[:,0], mask=pid_mask.squeeze(1), norm=True).item()) # the mean 
        valid_pid_mse_list[pid].append(mse_loss(output.squeeze(1), step_gt[:,0], mask=pid_mask.squeeze(1), norm=True).item()) # the mean
        valid_pid_num_list[pid].append(pid_mask.squeeze(1).sum().item())
    # compute the macro mae and rmse
    valid_macro_mae_list = []
    valid_macro_rmse_list = []
    for pid in range(len(pid_data)):
        valid_macro_mae = np.array(valid_pid_mae_list[pid]) * np.array(valid_pid_num_list[pid])
        valid_macro_mae = valid_macro_mae.sum() / np.array(valid_pid_num_list[pid]).sum()
        valid_macro_mae_list.append(valid_macro_mae)
        valid_macro_rmse = np.array(valid_pid_mse_list[pid]) * np.array(valid_pid_num_list[pid])
        valid_macro_rmse = valid_macro_rmse.sum() / np.array(valid_pid_num_list[pid]).sum()
        valid_macro_rmse_list.append(np.sqrt(valid_macro_rmse))
    valid_macro_mae = np.mean(valid_macro_mae_list)
    valid_macro_rmse = np.mean(valid_macro_rmse_list)

    ## Test process ##
    print(f"split {split_idx} | begin testing ...")
    
    test_dataset = MiceDataset(pid_data, split_idx, "test", ks, pad_full_weeks)
        
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
    
    output = imp.transform(input_feat)
    # get the unnormalized step rate
    output = output[:, -1][..., None] * sr_std + sr_mean # [N, 1]
    output = lower_upper_bound_func(output, 0.0, max_sr)
    # get the step counts
    output = output * step_gt[:, 1][..., None]
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
            
    return valid_micro_mae, valid_micro_rmse, valid_macro_mae, valid_macro_rmse, \
           test_micro_mae, test_micro_rmse, test_macro_mae, test_macro_rmse, \
           overall_time

def main():
    # get arguments
    args = get_args()
    num_split = args.num_split 
    mice_iter = args.mice_iter 
    epochs = args.epochs
    lr = args.lr
    batch_size = args.batch_size 
    kh, kw = args.kh, args.kw
    seed = args.seed
    store_model = args.store_model
    # for num_parts, please change it in the default function argument
    
    split_list = list(range(num_split))
    mice_iter_list = [mice_iter] * num_split
    epochs_list = [epochs] * num_split
    batch_size_list = [batch_size] * num_split 
    lr_list = [lr] * num_split
    seed_list = [seed] * num_split
    store_model_list = [store_model] * num_split

    # build folder to store the loss history and the best model
    new_dir(f"./results")
    # OUT_PATH = f"./results/mice_{args.mice_iter}_{args.epochs}_{args.batch_size}_{args.lr}"
    # new_dir(OUT_PATH)

    # write the statistics into the file
    # file_obj = open(args.output_file, "w")
    timestamp = datetime.now().strftime('%m%d.%H%M%S')

    file_obj = open(f"./results/{timestamp}_mice_{mice_iter}_{epochs}_{batch_size}_{lr}_{seed}.txt", "w")

    valid_micro_mae_list = []
    valid_macro_mae_list = []
    valid_micro_rmse_list = []
    valid_macro_rmse_list = []

    test_micro_mae_list = []
    test_macro_mae_list = []
    test_micro_rmse_list = []
    test_macro_rmse_list = []

    overall_time_list = []

    # for i in range(args.num_split):
        # start_time = time.time()
        # valid_micro_mae, valid_micro_rmse, valid_macro_mae, valid_macro_rmse, \
        #    test_micro_mae, test_micro_rmse, test_macro_mae, test_macro_rmse = main(args, i)
        # print(f"running time for one split is: {(time.time() - start_time):.2f} seconds")
        # file_obj.write(f"running time for one split is: {(time.time() - start_time):.2f} seconds\n")
    
    with multiprocessing.Pool() as pool:
        for results in pool.starmap(run_mice, list(zip(split_list, mice_iter_list, epochs_list, batch_size_list, lr_list, seed_list, store_model_list))):
 
            valid_micro_mae_list.append(results[0])
            valid_macro_mae_list.append(results[2])
            valid_micro_rmse_list.append(results[1])
            valid_macro_rmse_list.append(results[3])

            test_micro_mae_list.append(results[4])
            test_macro_mae_list.append(results[6])
            test_micro_rmse_list.append(results[5])
            test_macro_rmse_list.append(results[7])

            overall_time_list.append(results[8])
    
    for i in range(num_split):
        file_obj.write(f"split {i} | run time: {overall_time_list[i]:.2f} seconds\n")
    
    file_obj.write("\n")

    for i in range(num_split):
        #print(f"split {i} | best_epoch: {best_epoch} | best_val_mae: {best_val_nll:.2f} | test_mae_best_epoch: {hist_dict['mae']['test_loss'][best_epoch]:.2f}")
        file_obj.write(f"split {i} | valid_micro_mae {valid_micro_mae_list[i]:.2f} | test_micro_mae: {test_micro_mae_list[i]:.2f}\n")

    # MICRO MAE loss
    valid_micro_mae_mean = np.mean(valid_micro_mae_list)
    valid_micro_mae_std = np.std(valid_micro_mae_list)
    test_micro_mae_mean = np.mean(test_micro_mae_list)
    test_micro_mae_std = np.std(test_micro_mae_list)

    file_obj.write('\n')
    file_obj.write('MICRO MAE Statistics\n')
    file_obj.write("-"*50 + "\n")
    file_obj.write(f"valid | mean: {valid_micro_mae_mean:.2f} | std: {valid_micro_mae_std:.2f}\n")
    file_obj.write(f"test  | mean: {test_micro_mae_mean:.2f} | std: {test_micro_mae_std:.2f}\n")
    file_obj.write(f"diff of valid and test mean: {np.abs(valid_micro_mae_mean-test_micro_mae_mean):.2f}\n")

    # MACRO MAE
    file_obj.write("\n")
    for i in range(num_split):
        file_obj.write(f"split {i} | valid_macro_mae: {valid_macro_mae_list[i]:.2f} | test_macro_mae: {test_macro_mae_list[i]:.2f}\n")
    file_obj.write("\n")
    file_obj.write(f"MACRO MAE Best Epoch Statistics\n")
    file_obj.write("-" * 50 + "\n")
    file_obj.write(f"valid | mean: {np.mean(valid_macro_mae_list):.2f} | std: {np.std(valid_macro_mae_list):.2f}\n")
    file_obj.write(f"test  | mean: {np.mean(test_macro_mae_list):.2f} | std: {np.std(test_macro_mae_list):.2f}\n")
    
    # MICRO RMSE 
    file_obj.write("\n")
    for i in range(num_split):
        file_obj.write(f"split {i} | valid_micro_rmse: {valid_micro_rmse_list[i]:.2f} | test_micro_rmse: {test_micro_rmse_list[i]:.2f}\n")
    file_obj.write("\n")
    file_obj.write(f"MICRO RMSE Best Epoch Statistics\n")
    file_obj.write("-" * 50 + "\n")
    file_obj.write(f"valid | mean: {np.mean(valid_micro_rmse_list):.2f} | std: {np.std(valid_micro_rmse_list):.2f}\n")
    file_obj.write(f"test  | mean: {np.mean(test_micro_rmse_list):.2f} | std: {np.std(test_micro_rmse_list):.2f}\n")
    
    # MACRO RMSE
    file_obj.write("\n")
    for i in range(num_split):
        file_obj.write(f"split {i} | valid_macro_rmse: {valid_macro_rmse_list[i]:.2f} | test_macro_rmse: {test_macro_rmse_list[i]:.2f}\n")
    file_obj.write("\n")
    file_obj.write(f"MACRO RMSE Best Epoch Statistics\n")
    file_obj.write("-" * 50 + "\n")
    file_obj.write(f"valid | mean: {np.mean(valid_macro_rmse_list):.2f} | std: {np.std(valid_macro_rmse_list):.2f}\n")
    file_obj.write(f"test  | mean: {np.mean(test_macro_rmse_list):.2f} | std: {np.std(test_macro_rmse_list):.2f}\n")
    
    file_obj.close()

if __name__ == "__main__":
    main()
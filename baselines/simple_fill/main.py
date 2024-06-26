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
from baselines.simple_fill import model


def get_micro_macro_test_performance(df_cohort, num_split=10, fill_method="zero_fill", top_rows=None, verbose=True, file_obj=None):

    if top_rows is not None:
        df_cohort = df_cohort.head(top_rows)
        
    test_loss_list = []
    for pid in tqdm(df_cohort.index.to_list()):
        # read in the data
        df_exp, _, _ = data_utils.get_hourly_data(pid, num_split=num_split, conv_feat=False)
        # get the fill method
        if fill_method == "zero_fill":
            fill_func = model.zero_fill
        elif fill_method == "participant_average_mean":
            fill_func = model.participant_average_mean
        elif fill_method == "dayweek_average_mean":
            fill_func = model.dayweek_average_mean
        elif fill_method == "hour_average_mean":
            fill_func = model.hour_average_mean
        elif fill_method == "dayweek_hour_average_mean":
            fill_func = model.dayweek_hour_average_mean
        elif fill_method == "participant_mean":
            fill_func = model.participant_mean
        elif fill_method == "dayweek_mean":
            fill_func = model.dayweek_mean
        elif fill_method == "hour_mean":
            fill_func = model.hour_mean
        elif fill_method == "dayweek_hour_mean":
            fill_func = model.dayweek_hour_mean
        elif fill_method == "participant_median":
            fill_func = model.participant_median
        elif fill_method == "dayweek_median":
            fill_func = model.dayweek_median
        elif fill_method == "hour_median":
            fill_func = model.hour_median
        elif fill_method == "dayweek_hour_median":
            fill_func = model.dayweek_hour_median
        elif fill_method == "forward_filling":
            fill_func = model.forward_filling
        elif fill_method == "backward_filling":
            fill_func = model.backward_filling
        elif fill_method == "avg_forward_backward_fill":
            fill_func = model.avg_forward_backward_fill
        else:
            raise NameError(f"{fill_method} is not implemented!")
        
        # compute the test loss for each participant each split
        test_loss = fill_func(df_exp, num_splits=num_split, verbose=False)
        test_loss_list.append(test_loss)
        
    file_obj.write(f"Fill Method: {fill_method}\n")
    file_obj.write("*" * 100 + "\n")
    file_obj.write("\n")
        
    # macro average
    file_obj.write("MACRO STATISTICS\n")
    file_obj.write("-" * 100 + "\n")

    macro_mae_list = []
    macro_rmse_list = [] 
    for split_idx in range(num_split):
        mae_list_split = []
        rmse_list_split = []
        for test_loss_dict in test_loss_list:
            # mae
            mae_list_split.append(test_loss_dict["mae"][split_idx])
            # rmse
            rmse_list_split.append(test_loss_dict["rmse"][split_idx])
        macro_mae_list.append(np.mean(mae_list_split))
        macro_rmse_list.append(np.mean(rmse_list_split))
        if verbose:
            file_obj.write(f"split {split_idx} | macro mae: {np.mean(mae_list_split):.2f} (std: {np.std(mae_list_split):.2f}) | macro rmse: {np.mean(rmse_list_split):.2f} (std: {np.std(rmse_list_split):.2f})\n")

    file_obj.write("\n")
    file_obj.write(f"{num_split} splits | mean macro mae: {np.mean(macro_mae_list):.2f} (std: {np.std(macro_mae_list):.2f})\n")
    file_obj.write(f"{num_split} splits | mean macro rmse: {np.mean(macro_rmse_list):.2f} (std: {np.std(macro_rmse_list):.2f})\n")
    file_obj.write("\n")
    file_obj.write("\n")

    # micro average
    file_obj.write("MICRO STATISTICS\n")
    file_obj.write("-" * 100 + '\n')

    micro_mae_list = []
    micro_rmse_list = []
    for split_idx in range(num_split):
        mae_list_split = []
        mse_list_split = []  # note we cannot use rmse here
        num_hourly_blocks_split = []
        for test_loss_dict in test_loss_list:
            # mae
            mae_list_split.append(test_loss_dict["mae"][split_idx])
            # mse
            mse_list_split.append(test_loss_dict["mse"][split_idx])
            # num of valid hourly blocks
            num_hourly_blocks_split.append(test_loss_dict["num_hourly_blocks"][split_idx])

        micro_mae = np.sum(np.array(mae_list_split) * np.array(num_hourly_blocks_split)) / np.sum(num_hourly_blocks_split)
        micro_mse = np.sum(np.array(mse_list_split) * np.array(num_hourly_blocks_split)) / np.sum(num_hourly_blocks_split)
        micro_mae_list.append(micro_mae)
        micro_rmse_list.append(np.sqrt(micro_mse))
        if verbose:
            file_obj.write(f"split {split_idx} | micro mae: {micro_mae:.2f} | micro rmse: {np.sqrt(micro_mse):.2f}\n")

    file_obj.write('\n')
    file_obj.write(f"mean micro mae: {np.mean(micro_mae_list):.2f} (std: {np.std(micro_mae_list):.2f})\n")
    file_obj.write(f"mean micro rmse: {np.mean(micro_rmse_list):.2f} (std: {np.std(micro_rmse_list):.2f})\n")
    file_obj.write('\n')
    file_obj.write('\n')

if __name__ == "__main__":
    # pull files
    data_utils.pull_file(data_utils.DATA_FILE)
    data_utils.pull_file(data_utils.START_END_FILE)
    data_utils.pull_file(data_utils.SHIFT_FILE)
    data_utils.pull_file(data_utils.INFO_FILE)
    data_utils.pull_file(data_utils.INDEX_FILE)

    # pull the entire cohort we created (>7000 participants)
    data_utils.pull_file("all_stats_qual_part.parquet")

    # get the cohort
    df_start_end_day = pd.read_parquet(f"{data_utils.FILE_CACHE}/all_stats_qual_part.parquet")
    # sort the valid hourly blocks
    df_start_end_day = df_start_end_day.sort_values(by=["Valid_Hourly_Blocks", "total_valid_minutes"], ascending=[False, False])
    # get first 100 participants
    df_cohort = df_start_end_day.head(100)

    train_utils.new_dir("./results")

    # write the statistics into the file
    baseline_results_filename = "./results/baseline_results.txt"
    file_obj = open(baseline_results_filename, "w")

    # total valid hours
    file_obj.write(f'total number of valid hourly blocks {df_cohort["Valid_Hourly_Blocks"].sum()}\n')
    file_obj.write(f'total valid hours {df_cohort["total_valid_minutes"].sum()/60:.2f}\n')
    file_obj.write("\n")
    file_obj.write("\n")

    # get the performance
    fill_method_list = [
                        "zero_fill",
                        "forward_filling",
                        "backward_filling",
                        "avg_forward_backward_fill",
                        "participant_average_mean",
                        "dayweek_average_mean",
                        "hour_average_mean",
                        "dayweek_hour_average_mean",
                        "participant_mean",
                        "dayweek_mean",
                        "hour_mean",
                        "dayweek_hour_mean",
                        "participant_median",
                        "dayweek_median",
                        "hour_median",
                        "dayweek_hour_median",
                        ]
    

    for fill_method in fill_method_list:
        get_micro_macro_test_performance(df_cohort, 
                                         num_split=10, 
                                         fill_method=fill_method, 
                                         top_rows=None, 
                                         verbose=True,
                                         file_obj=file_obj)

    file_obj.close()

    # copy it into the google bucket
    os.system(f"gsutil -m cp {baseline_results_filename} {os.getenv('WORKSPACE_BUCKET')+'/data/'}")





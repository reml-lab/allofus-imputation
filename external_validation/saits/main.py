"""
This script is to evaluate the model of self attention with the LAPR as the feature
on the external validation set (both high and low miss rate)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
from tqdm import tqdm
from collections import OrderedDict

import sys
sys.path.append("../..")

from utils.data_utils import FILE_CACHE, pull_file
from utils.train_utils import new_dir, mse_loss, mae_loss, feature_padding
from baselines.saits.model import SelfAttnMultiLayerLAPR
from external_validation.saits.dataset import AllOfUsDatasetExtValid, BatchCollate
    
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
    parser.add_argument('--kw', type=int, default=15)
    parser.add_argument('--pad-full-weeks', action="store_true")
    parser.add_argument('--num-split', type=int, default=10)
    parser.add_argument('--if-high-miss', action="store_true")
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--d-k', type=int, default=2)
    parser.add_argument('--d-v', type=int, default=1)
    parser.add_argument('--if-regress', action="store_true")
    parser.add_argument('--lapr-rep', action="store_true")  # whether to use lapr feature to replace one hot vector
    parser.add_argument('--output-file', type=str, default=f'./output.txt')
    parser.add_argument('--pid-feat', action="store_true")  # to add participant indicator to the model as the feature
    parser.add_argument('--viz-attn', action="store_true") # wheter to store the attention weights
    parser.add_argument('--save-pred', action="store_true")  # whether to save prediction results

    parser.add_argument('--all-gpus', action="store_true")
    parser.add_argument('--gpu-id', type=int, default=0) 

    args = parser.parse_args()

    return args


### MAIN ###
def main(args, split_idx):
    
    ### MAIN ### 
    print(f"split {split_idx} begins ...")

    ## deal with the randomization ##
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    ## define the device ##
    if not args.all_gpus:
        device = torch.device(
            f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
        print(f"device: {device}")
    else:
        device = torch.device(
            f'cuda' if torch.cuda.is_available() else 'cpu'
        )
        print("We are using multiple GPUs ...")
        print(f"The number of GPUs: {torch.cuda.device_count()}")

    # get the data for the participant
    if args.if_high_miss:
        pull_file("pid_data_extvalid_high_miss.pkl")
        with open(f"{FILE_CACHE}/pid_data_extvalid_high_miss.pkl", "rb") as fin:
            pid_data = pickle.load(fin)
        # get the participant id list
        pull_file("df_cohort_extvalid_high_missrate.parquet")  # created by get_cohort_aaai.ipynb
        df_cohort = pd.read_parquet(f"{FILE_CACHE}/df_cohort_extvalid_high_missrate.parquet")
        pid_list = df_cohort.index.tolist()
    else:
        pull_file("pid_data_extvalid_low_miss.pkl")
        with open(f"{FILE_CACHE}/pid_data_extvalid_low_miss.pkl", "rb") as fin:
            pid_data = pickle.load(fin)
        # get the participant id list
        pull_file("df_cohort_extvalid_low_missrate.parquet")  # created by get_cohort_aaai.ipynb
        df_cohort = pd.read_parquet(f"{FILE_CACHE}/df_cohort_extvalid_low_missrate.parquet")
        pid_list = df_cohort.index.tolist()
    
    ks = (args.kh, args.kw)
    ctx_len = 72  # the window size on one side for the raw feature

    ########################################################
    # we only do the error analysis on the test set for now
    ########################################################
    
    # define the dataset and dataloader
    batch_collate = BatchCollate(ctx_len)
    ## test ##
    test_dataset = AllOfUsDatasetExtValid(pid_list, pid_data, split_idx, dataset="test", ks=ks, pad_full_weeks=args.pad_full_weeks, if_high_miss=args.if_high_miss)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=batch_collate, pin_memory=False)
    print(f"split {split_idx} | test | input_feat shape: {test_dataset.input_feat_pids.shape} | raw_feat shape: {test_dataset.lapr_feat_pids.shape}")
    
    ### define the model ###
    dp_rate = None
    
    model = SelfAttnMultiLayerLAPR( kernel_size=ks, 
                                    stride=1, 
                                    pad_full_weeks=args.pad_full_weeks, 
                                    conv_out_channels=1,
                                    d_k=args.d_k, 
                                    d_v=args.d_v, 
                                    if_regress=True, 
                                    dp_rate=dp_rate, 
                                    pid_feat=args.pid_feat, 
                                    lapr_rep=args.lapr_rep)
    
    # load the trained model parameters
    model_file_name = f"best_model_self_attn_multilayer_lapr_split_{split_idx}_seed_{args.seed}_dk_{args.d_k}_kh_9_kw_71"
    if args.pid_feat:
        model_file_name += "_with_pid"
    if not args.pad_full_weeks:
        model_file_name += "_same_day"
    if args.lapr_rep:
        model_file_name += "_lapr_rep"
    model_file_name += ".pth"
    
    pull_file(model_file_name) 
    ckpt = torch.load(f"{FILE_CACHE}/{model_file_name}", map_location="cpu")

    # remove "module." in the name of state_dict
    #from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in ckpt["state_dict"].items():
        name = k[7:] # remove `module.`
        ## used for ablation study ##
        if name == "rel_embed":
            if args.kh==9 and args.kw!=71:
                shift = (71-args.kw)//14
                v = v.reshape(9, 23)[:, shift:-shift].flatten()
            elif args.kh!=9 and args.kw==71:
                shift = (9-args.kh)//2
                v = v.reshape(9, 23)[shift:-shift, :].flatten()
        new_state_dict[name] = v
    # load params
    model.load_state_dict(new_state_dict)

    # load the model status
    # model.load_state_dict(ckpt["state_dict"])
    
    # load onto the device
    if args.all_gpus:
        model= nn.DataParallel(model)
    
    model.to(device)
    
    print("finish defining the model!")
    
    ### test process ###
    print("begin testing ...")
    with torch.no_grad():
        model.eval()
        
        # used to compute micro mae and mse
        test_total_mae = 0
        test_total_mse = 0
        test_total_num = 0
        
        # used to compute macro mae and mse
        test_pid_mae_list = [[] for _ in range(len(pid_data))]
        test_pid_mse_list = [[] for _ in range(len(pid_data))]
        test_pid_num_list = [[] for _ in range(len(pid_data))]

        p_attn_list = [] # store attetion score of all batches
        pid_ids_list = [] # store pid of each context window
        hour_list = [] # store hour of the center hourly block
        dayweek_list = [] # store day of the week of the center hourly block
        pred_list = [] # store the predicted step count
        gt_list = []  # store the groundtruth value for step counts
        
        for input_feat, lapr_feat, step_gt, max_sr, sr_mean, sr_std, pid_ids in tqdm(test_loader):
            
            ## ablation study ##
            lapr_feat = lapr_feat.reshape(lapr_feat.shape[0], 9, 23, lapr_feat.shape[-1])
            if args.kh==9 and args.kw!=71:
                shift = (71-args.kw)//14
                lapr_feat = lapr_feat[:, :, shift:-shift, :]
            elif args.kh!=9 and args.kw==71:
                shift = (9-args.kh)//2
                lapr_feat = lapr_feat[:, shift:-shift, :, :]
            lapr_feat = lapr_feat.reshape(lapr_feat.shape[0], -1, lapr_feat.shape[-1])
            ###### 

            input_feat = input_feat.to(device)
            lapr_feat = lapr_feat.to(device)
            step_gt = step_gt.to(device)
            max_sr = max_sr.to(device)
            sr_mean = sr_mean.to(device)
            sr_std = sr_std.to(device)
            pid_ids = pid_ids.to(device)
            
            if args.viz_attn:
                output, p_attn = model(input_feat, lapr_feat, max_sr, sr_mean, sr_std, pid_ids)
                p_attn_list.append(p_attn.cpu().numpy())
                # pid_ids_list.append(pid_ids.cpu().numpy())
                # dayweek_list.append(input_feat[:, 2, input_feat.shape[2]//2].cpu().numpy())
                # hour_list.append(input_feat[:, 3, input_feat.shape[2]//2].cpu().numpy())
            else:
                output = model(input_feat, lapr_feat, max_sr, sr_mean, sr_std, pid_ids)
            
            output = output * step_gt[:,1].unsqueeze(1)
            test_mae = mae_loss(output.squeeze(1), step_gt[:,0], mask=None, norm=True)
            test_mse = mse_loss(output.squeeze(1), step_gt[:,0], mask=None, norm=True)
            test_total_mae += test_mae * output.shape[0]
            test_total_mse += test_mse * output.shape[0]
            test_total_num += output.shape[0]
            
            if args.save_pred:
                pid_ids_list.append(pid_ids.cpu().numpy())
                dayweek_list.append(input_feat[:, 2, input_feat.shape[2]//2].cpu().numpy())
                hour_list.append(input_feat[:, 3, input_feat.shape[2]//2].cpu().numpy())
                pred_list.append(output.squeeze(1).cpu().numpy()) # store the predicted step count
                gt_list.append(step_gt[:,0].cpu().numpy())  # store the groundtruth value for step counts
            
            # macro mae and mse
            for pid in range(len(pid_data)):
                pid_mask = (pid_ids==pid).int()
                test_pid_mae_list[pid].append(mae_loss(output.squeeze(1), step_gt[:,0], mask=pid_mask.squeeze(1), norm=True).item()) # the mean 
                test_pid_mse_list[pid].append(mse_loss(output.squeeze(1), step_gt[:,0], mask=pid_mask.squeeze(1), norm=True).item()) # the mean
                test_pid_num_list[pid].append(pid_mask.squeeze(1).sum().item())

        # compute the macro mae and rmse
        test_macro_mae_list = []
        test_macro_rmse_list = []

        for pid in range(len(pid_data)):
            # mae
            test_macro_mae = np.array(test_pid_mae_list[pid]) * np.array(test_pid_num_list[pid])
            test_macro_mae = test_macro_mae.sum() / np.array(test_pid_num_list[pid]).sum()
            test_macro_mae_list.append(test_macro_mae)
            # rmse
            test_macro_rmse = np.array(test_pid_mse_list[pid]) * np.array(test_pid_num_list[pid])
            test_macro_rmse = test_macro_rmse.sum() / np.array(test_pid_num_list[pid]).sum()
            test_macro_rmse_list.append(np.sqrt(test_macro_rmse))

        test_macro_mae = np.mean(test_macro_mae_list)
        test_macro_rmse = np.mean(test_macro_rmse_list)
       
        best_epoch = ckpt["best_epoch"]
        
        test_micro_mae = test_total_mae.item() / test_total_num
        test_micro_rmse = np.sqrt(test_total_mse.item() / test_total_num)
        
        print(f"split: {split_idx} | epoch: {best_epoch} | test_micro_mae:  {test_micro_mae:.2f} | test_macro_mae: {test_macro_mae:.2f}")
        print(f"split: {split_idx} | epoch: {best_epoch} | test_micro_rmse: {test_micro_rmse:.2f} | test_macro_rmse: {test_macro_rmse:.2f}")
        
    if args.viz_attn:
        p_attn_bundle = {"p_attn": np.concatenate(p_attn_list, axis=0),
                         "pid_ids": np.concatenate(pid_ids_list, axis=0),
                         "hours": np.concatenate(hour_list, axis=0),
                         "dayweek": np.concatenate(dayweek_list, axis=0)}

        with open(f"{OUT_PATH}/p_attn_dict_split{split_idx}.pkl", "wb") as fout:
            pickle.dump(p_attn_bundle, fout)

    if args.save_pred:
        pred_bundle = {"pred": np.concatenate(pred_list, axis=0),
                       "gt": np.concatenate(gt_list, axis=0),
                       "pids": np.concatenate(pid_ids_list, axis=0),
                       "hours": np.concatenate(hour_list, axis=0),
                       "dayweek": np.concatenate(dayweek_list, axis=0)}
        with open(f"{OUT_PATH}/pred_results_dict_self_attn_multilayer_lapr_split{split_idx}.pkl", "wb") as fout:
            pickle.dump(pred_bundle, fout)

    print(f"split {split_idx} is finished!")
   
    return best_epoch, test_micro_mae, test_macro_mae, test_micro_rmse, test_macro_rmse, split_idx
    

if __name__ == "__main__":

    # get arguments
    args = get_args()

    # build folder to store the loss history and the best model
    new_dir(f"./results")
    OUT_PATH = f"./results/self_attn_multilayer_lapr_{args.kh}_{args.kw}_pred_results_extvalid"
    if args.if_high_miss:
        OUT_PATH += "_high_miss"
    else:
        OUT_PATH += "_low_miss"
    new_dir(OUT_PATH)
    error_dict = {"micro_mae":[], "macro_mae":[], "micro_rmse":[], "macro_rmse":[]}
    # write the statistics into the file
    file_obj = open(args.output_file, "w")
    
    for i in range(args.num_split):
        
        best_epoch, test_micro_mae, test_macro_mae, test_micro_rmse, test_macro_rmse, split_idx = main(args, i)
        
        error_dict["micro_mae"].append(test_micro_mae)
        error_dict["macro_mae"].append(test_macro_mae)
        error_dict["micro_rmse"].append(test_micro_rmse)
        error_dict["macro_rmse"].append(test_macro_rmse)

        file_obj.write(f"test | split: {i} | best_epoch: {best_epoch}\n")
        file_obj.write(f"test | split: {i} | micro mae: {test_micro_mae:.2f} | macro mae: {test_macro_mae:.2f}\n") 
        file_obj.write(f"test | split: {i} | micro rmse: {test_micro_rmse:.2f} | macro rmse: {test_macro_rmse:.2f}\n")
        file_obj.write("-" * 100 + '\n')
        file_obj.write("\n")
        
    file_obj.write("\n")
    file_obj.write(f"Test Error over {args.num_split} Splits\n")
    file_obj.write(f"Micro MAE mean(std): {np.mean(error_dict['micro_mae']):.2f} ({np.std(error_dict['micro_mae']):.2f})\n")
    file_obj.write(f"Macro MAE mean(std): {np.mean(error_dict['macro_mae']):.2f} ({np.std(error_dict['macro_mae']):.2f})\n")
    file_obj.write(f"Micro RMSE mean(std): {np.mean(error_dict['micro_rmse']):.2f} ({np.std(error_dict['micro_rmse']):.2f})\n")
    file_obj.write(f"Macro RMSE mean(std): {np.mean(error_dict['macro_rmse']):.2f} ({np.std(error_dict['macro_rmse']):.2f})\n")
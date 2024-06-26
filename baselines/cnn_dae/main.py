"""
Denoise Convolutional AutoEncoder Train and Eval Procedure
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

import sys
sys.path.append("../..")

from utils.data_utils import FILE_CACHE, pull_file
from utils.train_utils import new_dir, mse_loss, mae_loss, feature_padding
from baselines.cnn_dae.model import CNN_DAE
from baselines.cnn_dae.dataset import AllOfUsDataset
    
import warnings
warnings.filterwarnings("ignore")

def get_args():
    """
    parser the arguments to tune the models
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--save-last', action="store_true") # to save the model from the last epoch
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--kh', type=int, default=9)
    parser.add_argument('--kw', type=int, default=15)
    parser.add_argument('--pad-full-weeks', action="store_true")
    parser.add_argument('--num-split', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=800)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--nhr-coeff', type=float, default=0.0)
    parser.add_argument('--output-file', type=str, default=f'./output.txt')
    parser.add_argument('--all-gpus', action="store_true")
    parser.add_argument('--gpu-id', type=int, default=0) 

    args = parser.parse_args()

    return args

### MAIN ###
def main(args, split_idx):

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
    pull_file("pid_data.pkl")
    with open(f"{FILE_CACHE}/pid_data.pkl", "rb") as fin:
        pid_data = pickle.load(fin)
    ks = (args.kh, args.kw)

    ## Dataset and DataLoader ##
    train_dataset = AllOfUsDataset(pid_data, split_idx, dataset="train", ks=ks, pad_full_weeks=args.pad_full_weeks)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    
    valid_dataset = AllOfUsDataset(pid_data, split_idx, dataset="valid", ks=ks, pad_full_weeks=args.pad_full_weeks)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=args.batch_size, shuffle=False)
    
    test_dataset = AllOfUsDataset(pid_data, split_idx, dataset="test", ks=ks, pad_full_weeks=args.pad_full_weeks)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False)

    ## Model ##
    model = CNN_DAE()
    
    if args.all_gpus:
        model= nn.DataParallel(model)
    model.to(device)
    
    print("finish defining the model!")

    params = list(model.parameters())
    optimizer = optim.Adam(params, lr=args.lr)
    
    hist_dict = {"mae": {"train_loss":[], "valid_loss":[], "test_loss":[]},
                 "mse": {"valid_loss":[], "test_loss":[]},
                 "macro_mae": {"valid_loss":[], "test_loss":[]},
                 "macro_rmse": {"valid_loss":[], "test_loss":[]}}
     
    best_val_nll = np.inf
    best_epoch = 0

    for epoch in range(args.epochs):
        ## Train Process ##
        model.train()

        train_total_dae_loss = 0  # total training dae loss
        train_total_mae = 0  # total training mae loss for the center hourly block we need to impute
        train_total_num = 0  # total number of training instances 

        for idx, (input_feat, step_gt, part_mean_sr, max_sr, sr_mean, sr_std, pid_ids) in enumerate(train_loader):
            
            print(f"begin iter {idx} ...")

            input_feat = input_feat.to(device)
            step_gt = step_gt.to(device)
            part_mean_sr = part_mean_sr.to(device)
            max_sr = max_sr.to(device)
            sr_mean = sr_mean.to(device)
            sr_std = sr_std.to(device) 

            dae_input_feat = input_feat.clone()
            # we also need to get the output to train the autoencoder
            dae_gt = input_feat.clone()
            dae_gt_value = dae_gt[:, [0,1], :]
            dae_gt_mask = dae_gt[:, 3, :]#.unsqueeze(1) # only evaluate on the all the training points [bs, L]
            # add noise (fill the center step rate and heart rate and other masked blocks as zeros)
            dae_input_mask = dae_input_feat[:,2,:].unsqueeze(1)
            # mask out the center hourly block
            dae_input_mask[:, :, dae_input_mask.shape[-1]//2] = 0
            # get the noisy data
            dae_input_feat_corr = dae_input_feat[:, [0,1], :] # get step rate and heart rate
            dae_input_feat_corr = dae_input_feat_corr * dae_input_mask

            # dae train     
            optimizer.zero_grad()
            output = model(dae_input_feat_corr, max_sr, sr_mean, sr_std) # forward pass shape: [bs, L, 2]  
            # we only evaluate on the hourly blocks belonging to the training dataset
            nsr_loss = torch.abs(dae_gt_value[:, 0, :] - output[:, 0, :])[dae_gt_mask==1].mean()
            nhr_loss = torch.abs(dae_gt_value[:, 1, :] - output[:, 1, :])[dae_gt_mask==1].mean()
            train_dae_loss = nsr_loss + args.nhr_coeff * nhr_loss
            train_total_dae_loss += train_dae_loss * output.shape[0]
            train_total_num += output.shape[0]
            train_dae_loss.backward() # backward pass
            optimizer.step()
            
            with torch.no_grad():
                # we need to assess on the step count level
                output_step_rate = output[:,0, output.shape[2]//2].unsqueeze(1) * sr_std + sr_mean
                output_step_count = output_step_rate.squeeze(1) * step_gt[:,1]
                train_mae_loss = torch.abs(output_step_count - step_gt[:,0]).mean()
                # train_mse = ((output_step_count - step_gt[:,0])**2).mean()
                train_total_mae += train_mae_loss * output.shape[0]

        ## Valid Process ##
        print("begin validation ...")
        with torch.no_grad():
            model.eval()
            
            # used to compute micro mae and mse
            valid_total_mae = 0
            valid_total_mse = 0
            valid_total_num = 0
            
            # used to compute macro mae and mse
            valid_pid_mae_list = [[] for _ in range(len(pid_data))]
            valid_pid_mse_list = [[] for _ in range(len(pid_data))]
            valid_pid_num_list = [[] for _ in range(len(pid_data))]
            
            for input_feat, step_gt, part_mean_sr, max_sr, sr_mean, sr_std, pid_ids in valid_loader:
                
                input_feat = input_feat.to(device)
                step_gt = step_gt.to(device)
                part_mean_sr = part_mean_sr.to(device)
                max_sr = max_sr.to(device)
                sr_mean = sr_mean.to(device)
                sr_std = sr_std.to(device)
                pid_ids = pid_ids.to(device)

                dae_input_feat = input_feat.clone()
                # we also need to get the output to train the autoencoder
                dae_gt = input_feat.clone()
                dae_gt_value = dae_gt[:, [0,1], :]
                dae_gt_mask = dae_gt[:, 3, :]#.unsqueeze(1) # only evaluate on the all the training points [bs, L]
                # add noise (fill the center step rate and heart rate and other masked blocks as zeros)
                dae_input_mask = dae_input_feat[:,2,:].unsqueeze(1)
                # mask out the center hourly block
                dae_input_mask[:, :, dae_input_mask.shape[-1]//2] = 0
                # get the noisy data
                dae_input_feat_corr = dae_input_feat[:, [0,1], :] # get step rate and heart rate
                dae_input_feat_corr = dae_input_feat_corr * dae_input_mask
                
                output = model(dae_input_feat_corr, max_sr, sr_mean, sr_std) # forward pass shape: [bs, L, 2]  
                # we need to assess on the step count level
                output_step_rate = output[:,0, output.shape[2]//2].unsqueeze(1) * sr_std + sr_mean
                output_step_count = output_step_rate.squeeze(1) * step_gt[:,1]
                
                # micro mae and mse
                valid_mae = torch.abs(output_step_count - step_gt[:,0]).mean()
                valid_mse = ((output_step_count - step_gt[:,0])**2).mean()
                valid_total_mae += valid_mae * output.shape[0]
                valid_total_mse += valid_mse * output.shape[0]
                valid_total_num += output.shape[0]
                
                # macro mae and mse
                for pid in range(len(pid_data)):
                    pid_mask = (pid_ids==pid).int()
                    valid_pid_mae_list[pid].append(mae_loss(output_step_count, step_gt[:,0], mask=pid_mask.squeeze(1), norm=True).item()) # the mean 
                    valid_pid_mse_list[pid].append(mse_loss(output_step_count, step_gt[:,0], mask=pid_mask.squeeze(1), norm=True).item()) # the mean
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

        ### Test Process ###
        print("begin test ...")
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
            
            for input_feat, step_gt, part_mean_sr, max_sr, sr_mean, sr_std, pid_ids in test_loader:
                
                input_feat = input_feat.to(device)
                step_gt = step_gt.to(device)
                part_mean_sr = part_mean_sr.to(device)
                max_sr = max_sr.to(device)
                sr_mean = sr_mean.to(device)
                sr_std = sr_std.to(device)
                pid_ids = pid_ids.to(device)

                dae_input_feat = input_feat.clone()
                # we also need to get the output to train the autoencoder
                dae_gt = input_feat.clone()
                dae_gt_value = dae_gt[:, [0,1], :]
                dae_gt_mask = dae_gt[:, 3, :]#.unsqueeze(1) # only evaluate on the all the training points [bs, L]
                # add noise (fill the center step rate and heart rate and other masked blocks as zeros)
                dae_input_mask = dae_input_feat[:,2,:].unsqueeze(1)
                # mask out the center hourly block
                dae_input_mask[:, :, dae_input_mask.shape[-1]//2] = 0
                # get the noisy data
                dae_input_feat_corr = dae_input_feat[:, [0,1], :] # get step rate and heart rate
                dae_input_feat_corr = dae_input_feat_corr * dae_input_mask
                
                output = model(dae_input_feat_corr, max_sr, sr_mean, sr_std) # forward pass shape: [bs, L, 2]  
                # we need to assess on the step count level
                output_step_rate = output[:,0, output.shape[2]//2].unsqueeze(1) * sr_std + sr_mean
                output_step_count = output_step_rate.squeeze(1) * step_gt[:,1]
                
                # micro mae and mse
                test_mae = torch.abs(output_step_count - step_gt[:,0]).mean()
                test_mse = ((output_step_count - step_gt[:,0])**2).mean()
                test_total_mae += test_mae * output.shape[0]
                test_total_mse += test_mse * output.shape[0]
                test_total_num += output.shape[0]

                # macro mae and mse
                for pid in range(len(pid_data)):
                    pid_mask = (pid_ids==pid).int()
                    test_pid_mae_list[pid].append(mae_loss(output_step_count, step_gt[:,0], mask=pid_mask.squeeze(1), norm=True).item()) # the mean 
                    test_pid_mse_list[pid].append(mse_loss(output_step_count, step_gt[:,0], mask=pid_mask.squeeze(1), norm=True).item()) # the mean
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
        
        if args.verbose:
            train_avg_dae_loss = train_total_dae_loss.item() / train_total_num
            train_avg_mae = train_total_mae.item() / train_total_num
            valid_avg_mae = valid_total_mae.item() / valid_total_num
            test_avg_mae = test_total_mae.item() / test_total_num
            print(f"split: {split_idx} | epoch: {epoch} | train_dae: {train_avg_dae_loss:.4f} | train_mae: {train_avg_mae:.4f} | valid_mae: {valid_avg_mae:.4f} | valid_macro_mae: {valid_macro_mae:.4f}" \
                  f"| test_mae: {test_avg_mae:.4f} | test_macro_mae: {test_macro_mae:.4f}")

        hist_dict["mae"]["train_loss"].append(train_total_mae.item() / train_total_num)
        hist_dict["mae"]["valid_loss"].append(valid_total_mae.item() / valid_total_num)
        hist_dict["mae"]["test_loss"].append(test_total_mae.item() / test_total_num)
        
        hist_dict["mse"]["valid_loss"].append(valid_total_mse.item() / valid_total_num)
        hist_dict["mse"]["test_loss"].append(test_total_mse.item() / test_total_num)

        hist_dict["macro_mae"]["valid_loss"].append(valid_macro_mae)
        hist_dict["macro_mae"]["test_loss"].append(test_macro_mae)

        hist_dict["macro_rmse"]["valid_loss"].append(valid_macro_rmse)
        hist_dict["macro_rmse"]["test_loss"].append(test_macro_rmse)
        
        if (valid_total_mae.item() / valid_total_num) < best_val_nll:
            best_val_nll = valid_total_mae.item() / valid_total_num
            best_epoch = epoch
            # store the states of the best model
            if args.save:
                model_file_name = f"{OUT_PATH}/best_model_cnn_dae_split_{split_idx}_seed_{args.seed}_kh_{args.kh}_kw_{args.kw}_nhr_{args.nhr_coeff}"
                if not args.pad_full_weeks:
                    model_file_name += "_same_day"
                model_file_name += ".pth"
                torch.save({
                        'args': args,
                        'best_epoch': best_epoch,
                        'best_val_nll': best_val_nll,
                        'state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        # 'scheduler_state_dict': scheduler.state_dict(),
                    }, model_file_name)


    # store the loss history
    if args.save:
        loss_file_name = f"{OUT_PATH}/loss_history_cnn_dae_split_{split_idx}_seed_{args.seed}_kh_{args.kh}_kw_{args.kw}_nhr_{args.nhr_coeff}"
        if not args.pad_full_weeks:
            model_file_name += "_same_day"
        loss_file_name += ".pkl"
        with open(loss_file_name, "wb") as fout:
            pickle.dump(hist_dict, fout)

    # store the last epoch model
    if args.save_last:
        model_file_name = f"{OUT_PATH}/last_model_cnn_dae_split_{split_idx}_seed_{args.seed}_kh_{args.kh}_kw_{args.kw}_nhr_{args.nhr_coeff}"
        if not args.pad_full_weeks:
            model_file_name += "_same_day"
        model_file_name += ".pth"
        torch.save({
                    'args': args,
                    'epoch': epoch,
                    'best_epoch': best_epoch,
                    'best_val_nll': best_val_nll,
                    'state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    # 'scheduler_state_dict': scheduler.state_dict(),
                }, model_file_name)

    print(f"split {split_idx} is finished!")
            
    # this is for the case we don't have the gpu
    return hist_dict, best_epoch, best_val_nll, split_idx


if __name__ == "__main__":

    args = get_args()

    # build folder to store the loss history and the best model
    new_dir(f"./results")
    OUT_PATH = f"./results/cnn_dae_{args.nhr_coeff}"
    new_dir(OUT_PATH)

    # train and evaluate the model for each split
    hist_dict_dict = {}
    best_epoch_dict = {}
    train_best_epoch_dict = {}
    val_best_epoch_dict = {}
    test_best_epoch_dict = {}

    # write the statistics into the file
    file_obj = open(args.output_file, "w")

    for i in range(args.num_split):
        start_time = time.time()
        hist_dict, best_epoch, best_val_nll, _ = main(args, i)
        print(f"train time for one split is: {(time.time() - start_time):.2f} seconds")
        hist_dict_dict[i] = hist_dict
        best_epoch_dict[i] = best_epoch
        train_best_epoch_dict[i] = hist_dict["mae"]['train_loss'][best_epoch]
        val_best_epoch_dict[i] = best_val_nll
        test_best_epoch_dict[i] = hist_dict["mae"]['test_loss'][best_epoch]
        #print(f"split {i} | best_epoch: {best_epoch} | best_val_mae: {best_val_nll:.2f} | test_mae_best_epoch: {hist_dict['mae']['test_loss'][best_epoch]:.2f}")
        file_obj.write(f"split {i} | best_epoch: {best_epoch} | best_val_mae: {best_val_nll:.2f} | test_mae_best_epoch: {hist_dict['mae']['test_loss'][best_epoch]:.2f}\n")

    # MICRO MAE loss
    # get the mean and std of best models over 10 splits
    train_best_mean = np.mean([val for val in train_best_epoch_dict.values()])
    train_best_std = np.std([val for val in train_best_epoch_dict.values()])

    valid_best_mean = np.mean([val for val in val_best_epoch_dict.values()])
    valid_best_std = np.std([val for val in val_best_epoch_dict.values()])

    test_best_mean = np.mean([val for val in test_best_epoch_dict.values()])
    test_best_std = np.std([val for val in test_best_epoch_dict.values()])

    file_obj.write('\n')
    file_obj.write('MICRO MAE Best Epoch Statistics\n')
    file_obj.write("-"*50 + "\n")
    file_obj.write(f"train | mean: {train_best_mean:.2f} | std: {train_best_std:.2f}\n")
    file_obj.write(f"valid | mean: {valid_best_mean:.2f} | std: {valid_best_std:.2f}\n")
    file_obj.write(f"test  | mean: {test_best_mean:.2f} | std: {test_best_std:.2f}\n")
    file_obj.write(f"diff of valid and test mean: {np.abs(valid_best_mean-test_best_mean):.2f}\n")

    # MACRO MAE loss
    # get the mean and std of best models over 10 splits
    valid_best_macro_mae_list = []
    test_best_macro_mae_list = []

    file_obj.write("\n")
    for i in range(len(hist_dict_dict)):
        valid_best_macro_mae = hist_dict_dict[i]["macro_mae"]["valid_loss"][best_epoch_dict[i]]
        test_best_macro_mae = hist_dict_dict[i]["macro_mae"]["test_loss"][best_epoch_dict[i]]
        valid_best_macro_mae_list.append(valid_best_macro_mae)
        test_best_macro_mae_list.append(test_best_macro_mae)
        
        file_obj.write(f"split {i} | best_epoch: {best_epoch_dict[i]} | best_val_macro_mae: {valid_best_macro_mae:.2f} | test_macro_mae_best_epoch: {test_best_macro_mae:.2f}\n")
        
    file_obj.write("\n")
    file_obj.write(f"MACRO MAE Best Epoch Statistics\n")
    file_obj.write("-" * 50 + "\n")
    file_obj.write(f"valid | mean: {np.mean(valid_best_macro_mae_list):.2f} | std: {np.std(valid_best_macro_mae_list):.2f}\n")
    file_obj.write(f"test  | mean: {np.mean(test_best_macro_mae_list):.2f} | std: {np.std(test_best_macro_mae_list):.2f}\n")
     
    # MICRO MSE and RMSE loss
    # statistics for MSE and RMSE
    valid_best_mse_list = []
    test_best_mse_list = []
    test_best_rmse_list = []

    file_obj.write("\n")
    for i in range(len(hist_dict_dict)):
        valid_best_mse = hist_dict_dict[i]["mse"]["valid_loss"][best_epoch_dict[i]]
        test_best_mse = hist_dict_dict[i]["mse"]["test_loss"][best_epoch_dict[i]]
        valid_best_mse_list.append(valid_best_mse)
        test_best_mse_list.append(test_best_mse)
        test_best_rmse_list.append(np.sqrt(test_best_mse))
        
        file_obj.write(f"split {i} | best_epoch: {best_epoch_dict[i]} | best_val_mse: {valid_best_mse:.2f} | test_mse_best_epoch: {test_best_mse:.2f} | test_rmse_best_epoch: {np.sqrt(test_best_mse):.2f}\n")
        
    file_obj.write("\n")
    file_obj.write(f"MICRO MSE Best Epoch Statistics\n")
    file_obj.write("-" * 50 + "\n")
    file_obj.write(f"valid | mean: {np.mean(valid_best_mse_list):.2f} | std: {np.std(valid_best_mse_list):.2f}\n")
    file_obj.write(f"test  | mean: {np.mean(test_best_mse_list):.2f} | std: {np.std(test_best_mse_list):.2f}\n")
    file_obj.write('\n')
    file_obj.write(f"MICRO RMSE Best Epoch Statistics\n")
    file_obj.write(f"test | mean: {np.mean(test_best_rmse_list):.2f} | std: {np.std(test_best_rmse_list):.2f}\n")

    # MACRO RMSE
    # get the mean and std of best models over 10 splits
    valid_best_macro_rmse_list = []
    test_best_macro_rmse_list = []

    file_obj.write("\n")
    for i in range(len(hist_dict_dict)):
        valid_best_macro_rmse = hist_dict_dict[i]["macro_rmse"]["valid_loss"][best_epoch_dict[i]]
        test_best_macro_rmse = hist_dict_dict[i]["macro_rmse"]["test_loss"][best_epoch_dict[i]]
        valid_best_macro_rmse_list.append(valid_best_macro_rmse)
        test_best_macro_rmse_list.append(test_best_macro_rmse)
        
        file_obj.write(f"split {i} | best_epoch: {best_epoch_dict[i]} | best_val_macro_rmse: {valid_best_macro_rmse:.2f} | test_macro_rmse_best_epoch: {test_best_macro_rmse:.2f}\n")
        
    file_obj.write("\n")
    file_obj.write(f"MACRO RMSE Best Epoch Statistics\n")
    file_obj.write("-" * 50 + "\n")
    file_obj.write(f"valid | mean: {np.mean(valid_best_macro_rmse_list):.2f} | std: {np.std(valid_best_macro_rmse_list):.2f}\n")
    file_obj.write(f"test  | mean: {np.mean(test_best_macro_rmse_list):.2f} | std: {np.std(test_best_macro_rmse_list):.2f}\n")
    
    file_obj.close()


    ### plot the mean and std of the MAE loss over 10 splits ###
    # get the statistics for the plots
    train_loss_list = []
    valid_loss_list = []
    test_loss_list = []
    
    #for i in range(args.num_split):
    for i in hist_dict_dict.keys():
        train_loss_list.append(hist_dict_dict[i]["mae"]["train_loss"])
        valid_loss_list.append(hist_dict_dict[i]["mae"]["valid_loss"])
        test_loss_list.append(hist_dict_dict[i]["mae"]["test_loss"])
        

    train_mean = np.array(train_loss_list).mean(axis=0)
    train_std = np.array(train_loss_list).std(axis=0)

    valid_mean = np.array(valid_loss_list).mean(axis=0)
    valid_std = np.array(valid_loss_list).std(axis=0)

    test_mean = np.array(test_loss_list).mean(axis=0)
    test_std = np.array(test_loss_list).std(axis=0)

    # draw the plots
    w = 2.0
    epoch = np.arange(args.epochs) + 1

    fig, axs = plt.subplots(1, 2, figsize=(15, 5), sharex=True, sharey=True)
    axs[0].plot(epoch, valid_mean, c='darkblue', lw=w, label='valid_loss')
    axs[0].plot(epoch, test_mean, c='darkred', lw=w, label='test_loss')
    axs[0].fill_between(epoch, valid_mean-valid_std, valid_mean+valid_std, color='blue', alpha=0.5)
    axs[0].fill_between(epoch, test_mean-test_std, test_mean+test_std, color='red', alpha=0.5)
    axs[1].plot(epoch, train_mean, c='black', lw=w, label='train_loss')
    axs[1].fill_between(epoch, train_mean-train_std, train_mean+train_std, color='black', alpha=0.5)

    axs[0].set_ylabel("MAE Loss on Step Counts")
    axs[0].set_xlabel("Epochs")
    axs[1].set_xlabel("Epochs")

    axs[0].legend()
    axs[1].legend()
    axs[0].grid(linestyle='--')
    axs[1].grid(linestyle='--')

    plt.tight_layout()
    timestamp = datetime.now().strftime('%m%d.%H%M%S')
    plt.savefig(f"{OUT_PATH}/allofus_{timestamp}_cnn_dae_{args.lr}_{args.batch_size}_kh_{args.kh}_kw_{args.kw}_{args.nhr_coeff}_{args.seed}.png",\
                dpi=300)
    
    # copy the model from the VM disk to google cloud bucket
    if args.save:
        for split_idx in range(args.num_split):
            model_file_name = f"{OUT_PATH}/best_model_cnn_dae_split_{split_idx}_seed_{args.seed}_kh_{args.kh}_kw_{args.kw}_nhr_{args.nhr_coeff}"
            if not args.pad_full_weeks:
                model_file_name += "_same_day"
            model_file_name += ".pth"
            os.system(f"gsutil -m cp {model_file_name} {os.getenv('WORKSPACE_BUCKET')+'/data/'}")
        
    if args.save_last:
        for split_idx in range(args.num_split):
            model_file_name = f"{OUT_PATH}/last_model_cnn_dae_split_{split_idx}_seed_{args.seed}_kh_{args.kh}_kw_{args.kw}_nhr_{args.nhr_coeff}"
            if not args.pad_full_weeks:
                model_file_name += "_same_day"
            model_file_name += ".pth" 
            os.system(f"gsutil -m cp {model_file_name} {os.getenv('WORKSPACE_BUCKET')+'/data/'}")
        
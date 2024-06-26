"""
Train and Evaluation for USGAN (for one split)
(paper link: https://ojs.aaai.org/index.php/AAAI/article/view/17086)
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
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
from utils.train_utils import new_dir, mse_loss, mae_loss, make_forward_backward_data

from baselines.usgan.model import Generator, Discriminator
from dataset import AllOfUsDataset

import warnings
warnings.filterwarnings("ignore")

def get_args():
    """
    parser the arguments to tune the models
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--kh', type=int, default=9)
    parser.add_argument('--kw', type=int, default=15)
    parser.add_argument('--pad-full-weeks', action="store_true")
    parser.add_argument('--num-split', type=int, default=10)
    parser.add_argument('--gene-rnn-dim', type=int, default=16) # generator rnn hidden state dim
    parser.add_argument('--disc-rnn-dim', type=int, default=16) # discriminator rnn hidden state dim
    parser.add_argument('--hint-rate', type=float, default=0.8) # rate to retain missing indicators in the reminder matrix
    parser.add_argument('--alpha', type=float, default=5)  # weight before the adversarial loss when training the generator
    parser.add_argument('--epochs', type=int, default=800)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--output-file', type=str, default=f'./output.txt')

    parser.add_argument('--all-gpus', action="store_true")
    parser.add_argument('--gpu-id', type=int, default=0) 
    parser.add_argument('--split-idx', type=int, default=0)

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
    pull_file("pid_data.pkl")
    with open(f"{FILE_CACHE}/pid_data.pkl", "rb") as fin:
        pid_data = pickle.load(fin)
    
    # get the participant id list
    pull_file("df_cohort_top100.parquet")  # created by get_cohort_aaai.ipynb
    df_cohort = pd.read_parquet(f"{FILE_CACHE}/df_cohort_top100.parquet")
    pid_list = df_cohort.index.tolist()
    
    ks = (args.kh, args.kw)

    train_dataset = AllOfUsDataset(pid_data, split_idx, dataset="train", ks=ks, pad_full_weeks=args.pad_full_weeks)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    print(f"split {split_idx} | train | input_feat shape: {train_dataset.input_feat_pids.shape}")

    valid_dataset = AllOfUsDataset(pid_data, split_idx, dataset="valid", ks=ks, pad_full_weeks=args.pad_full_weeks)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=args.batch_size, shuffle=False)
    print(f"split {split_idx} | valid | input_feat shape: {valid_dataset.input_feat_pids.shape}")

    test_dataset = AllOfUsDataset(pid_data, split_idx, dataset="test", ks=ks, pad_full_weeks=args.pad_full_weeks)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False)
    print(f"split {split_idx} | test | input_feat shape: {test_dataset.input_feat_pids.shape}")

    ### define the model ###
    # generator
    generator = Generator(rnn_hid_size=args.gene_rnn_dim, device=device)
    # discriminator
    discriminator = Discriminator(rnn_hid_size=args.disc_rnn_dim, device=device)

    if args.all_gpus:
        generator= nn.DataParallel(generator)
        discriminator = nn.DataParallel(discriminator)
    
    generator.to(device)
    discriminator.to(device)
    
    print("finish defining the model!")

    gene_params = list(generator.parameters())
    disc_params = list(discriminator.parameters())
    
    optimizer_G = optim.Adam(gene_params, lr=args.lr)
    optimizer_D = optim.Adam(disc_params, lr=args.lr)

    # scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=100)

    # define loss functions
    adversarial_loss = torch.nn.BCEWithLogitsLoss(reduction='none')

    hist_dict = {"train_loss": {"generator":[], "discriminator":[]},
                 "mae": {"valid_loss":[], "test_loss":[]},
                 "mse": {"valid_loss":[], "test_loss":[]},
                 "macro_mae": {"valid_loss":[], "test_loss":[]},
                 "macro_rmse": {"valid_loss":[], "test_loss":[]}}

    best_val_nll = np.inf
    best_epoch = 0

    for epoch in range(args.epochs):
        # store the time for each epoch
        start_time_epoch = time.time()

        ## train process ##
        generator.train()
        discriminator.train()

        train_run_loss_G = 0  # total training loss for generator
        train_run_loss_D = 0  # total training loss for discriminator
        train_total_num_G = 0  # total number of training instances 
        train_total_num_D = 0  # total number of training instances

        for idx, (input_feat, step_gt, _, max_sr, sr_mean, sr_std, pid_ids) in enumerate(train_loader):
            
            print(f"begin iter {idx} ...")
            # ---------------
            # Train Generator 
            # ---------------
            
            print("Train Generator ...")

            optimizer_G.zero_grad()
            
            data_train = make_forward_backward_data(input_feat, max_sr, sr_mean, sr_std, ks)
            # note the generator loss includes both reconstruction loss and classification loss
            # so we don't use the optimizer here, just let the genertor do the forward pass
            if args.all_gpus:
                ret_train = generator.module.run_on_batch(data_train, optimizer=None, epoch=epoch)
            else:
                ret_train = generator.run_on_batch(data_train, optimizer=None, epoch=epoch)
            
            gene_data = ret_train["imputations"]  # [bs, seq_len, 2]
            miss_mask = data_train["forward"]["masks"].to(device)  # [bs, seq_len, 2]
            # classify by discriminator
            logits = discriminator(gene_data, miss_mask)
            fool_disc_label = torch.ones_like(miss_mask[:,:,0].unsqueeze(-1), device=device)  # [bs, seq_len, 1]
            g_loss = adversarial_loss(logits, fool_disc_label)
            # we only need to compute g_loss for the elements which have 0 in the missing mask
            g_loss = (g_loss[miss_mask[:,:,0].unsqueeze(-1)==0].sum() / (miss_mask[:,:,0].unsqueeze(-1)==0).int().sum())
            print(f"bce_loss: {g_loss.item():.4f} | brits_loss: {ret_train['loss'].item():.4f}")
            g_loss = args.alpha * g_loss + ret_train["loss"]  # 5 is default in the paper
            
            g_loss.backward()
            optimizer_G.step()

            train_run_loss_G += g_loss.item()
            train_total_num_G += 1

            # -------------------
            # Train Discriminator
            # -------------------
            # update discriminator 5 times after generator is updated
            print("Train Discriminator ...")
            #for _ in range(5):
            for _ in range(1):
                optimizer_D.zero_grad()
                
                # note we need to detach the generated data
                # since we don't want to train generator at this stage
                d_loss = adversarial_loss(discriminator(gene_data.detach(), miss_mask), miss_mask[:,:,0].unsqueeze(-1))
                d_loss = d_loss.sum() / (miss_mask.shape[0] * miss_mask.shape[1])
                
                d_loss.backward()
                optimizer_D.step()

                train_run_loss_D += d_loss.item()
                train_total_num_D += 1

        ### Valid Process ###
        print("begin validation ...")
        with torch.no_grad():
            # we don't need discriminator here
            generator.eval()
            
            # used to compute micro mae and mse
            valid_total_mae = 0
            valid_total_mse = 0
            valid_total_num = 0
            
            # used to compute macro mae and mse
            valid_pid_mae_list = [[] for _ in range(len(pid_data))]
            valid_pid_mse_list = [[] for _ in range(len(pid_data))]
            valid_pid_num_list = [[] for _ in range(len(pid_data))]

            for input_feat, step_gt, _, max_sr, sr_mean, sr_std, pid_ids in valid_loader:
                
                pid_ids = pid_ids.to(device)
                step_gt = step_gt.to(device)
                
                data_valid = make_forward_backward_data(input_feat, max_sr, sr_mean, sr_std, ks)
                if args.all_gpus:
                    ret_valid = generator.module.run_on_batch(data_valid, None)
                else:
                    ret_valid = generator.run_on_batch(data_valid, None)

                output = ret_valid["imputations"]  # [bs, 207, 2]
                # get the nsr for the center hourly block
                output = output[:, output.shape[1]//2, 0].unsqueeze(1)  # [bs, 1]
                # unnormalize it
                output = output * sr_std.to(device) + sr_mean.to(device)
                # step rate --> step count
                output = output * step_gt[:,1].unsqueeze(1)
                # micro mae and mse
                valid_mae = mae_loss(output.squeeze(1), step_gt[:,0], mask=None, norm=True)
                valid_mse = mse_loss(output.squeeze(1), step_gt[:,0], mask=None, norm=True)
                valid_total_mae += valid_mae * output.shape[0]
                valid_total_mse += valid_mse * output.shape[0]
                valid_total_num += output.shape[0]
                
                # macro mae and mse
                for pid in range(len(pid_data)):
                    pid_mask = (pid_ids==pid).int()
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

        # adjust the scheduler based on valid_mae
        # scheduler.step(valid_mae)

        ### Test Process ###
        print("begin testing ...")
        with torch.no_grad():
            # we don't need discriminator here
            generator.eval()

            # used to compute micro mae and mse
            test_total_mae = 0
            test_total_mse = 0
            test_total_num = 0

            # used to compute macro mae and mse
            test_pid_mae_list = [[] for _ in range(len(pid_data))]
            test_pid_mse_list = [[] for _ in range(len(pid_data))]
            test_pid_num_list = [[] for _ in range(len(pid_data))]
            
            for input_feat, step_gt, _, max_sr, sr_mean, sr_std, pid_ids in test_loader:
                
                step_gt = step_gt.to(device)
                pid_ids = pid_ids.to(device)

                data_test = make_forward_backward_data(input_feat, max_sr, sr_mean, sr_std, ks)
                if args.all_gpus:
                    ret_test = generator.module.run_on_batch(data_test, None)
                else:
                    ret_test = generator.run_on_batch(data_test, None)
                
                output = ret_test["imputations"]  # [bs, 207, 2]
                # get the nsr for the center hourly block
                output = output[:, output.shape[1]//2, 0].unsqueeze(1)  # [bs, 1]
                # unnormalize it
                output = output * sr_std.to(device) + sr_mean.to(device)
                # step rate --> step count
                output = output * step_gt[:,1].unsqueeze(1)
                
                test_mae = mae_loss(output.squeeze(1), step_gt[:,0], mask=None, norm=True)
                test_mse = mse_loss(output.squeeze(1), step_gt[:,0], mask=None, norm=True)
                test_total_mae += test_mae * output.shape[0]
                test_total_mse += test_mse * output.shape[0]
                test_total_num += output.shape[0]
                
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
                test_macro_mae = np.array(test_pid_mae_list[pid]) * np.array(test_pid_num_list[pid])
                test_macro_mae = test_macro_mae.sum() / np.array(test_pid_num_list[pid]).sum()
                test_macro_mae_list.append(test_macro_mae)
                test_macro_rmse = np.array(test_pid_mse_list[pid]) * np.array(test_pid_num_list[pid])
                test_macro_rmse = test_macro_rmse.sum() / np.array(test_pid_num_list[pid]).sum()
                test_macro_rmse_list.append(np.sqrt(test_macro_rmse))
                
            test_macro_mae = np.mean(test_macro_mae_list)
            test_macro_rmse = np.mean(test_macro_rmse_list)

        ### print results ###
        if args.verbose:
            total_time_epoch = time.time() - start_time_epoch
            train_avg_loss_G = train_run_loss_G / train_total_num_G  # generator loss
            train_avg_loss_D = train_run_loss_D / train_total_num_D  # discriminator loss  
            valid_avg_mae = valid_total_mae.item() / valid_total_num  # mae loss of step count in the center hourly block
            test_avg_mae = test_total_mae.item() / test_total_num  # mae loss of step count in the center hourly block
            print(f"split: {split_idx} | epoch: {epoch} | train_gene_loss: {train_avg_loss_G:.4f} | train_disc_loss: {train_avg_loss_D:.4f}"
                  f"| valid_mae: {valid_avg_mae:.4f} | valid_macro_mae: {valid_macro_mae:.4f}" \
                  f"| test_mae: {test_avg_mae:.4f} | test_macro_mae: {test_macro_mae:.4f}"\
                  f"| epoch_time: {total_time_epoch:.2f} seconds")
            
        hist_dict["train_loss"]["generator"].append(train_avg_loss_G)
        hist_dict["train_loss"]["discriminator"].append(train_avg_loss_D)
        
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
                model_file_name = f"{OUT_PATH}/best_model_usgan_split_{split_idx}_seed_{args.seed}_gene_{args.gene_rnn_dim}_disc_{args.disc_rnn_dim}"
                if not args.pad_full_weeks:
                    model_file_name += "_same_day"
                model_file_name += ".pth"
                torch.save({
                        'args': args,
                        'best_epoch': best_epoch,
                        'best_val_nll': best_val_nll,
                        'generator_state_dict': generator.state_dict(),
                        'discriminator_state_dict': discriminator.state_dict(),
                        'generator_optimizer_state_dict': optimizer_G.state_dict(),
                        'discriminator_state_dict': optimizer_D.state_dict()
                        # 'scheduler_state_dict': scheduler.state_dict(),
                    }, model_file_name)

    # store the loss history
    if args.save:
        loss_file_name = f"{OUT_PATH}/loss_history_usgan_split_{split_idx}_seed_{args.seed}_gene_{args.gene_rnn_dim}_disc_{args.disc_rnn_dim}"
        if not args.pad_full_weeks:
            loss_file_name += "_same_day"
        loss_file_name += ".pkl"
        with open(loss_file_name, "wb") as fout:
            pickle.dump(hist_dict, fout)

    print(f"split {split_idx} is finished!")
            
    # this is for the case we don't have the gpu
    return hist_dict, best_epoch, best_val_nll, split_idx


if __name__ == "__main__":

    # get arguments
    args = get_args()

    # build folder to store the loss history and the best model
    new_dir(f"./results")
    OUT_PATH = f"./results/usgan_gene_{args.gene_rnn_dim}_disc_{args.disc_rnn_dim}_hint_{args.hint_rate}_alpha_{args.alpha}"
    new_dir(OUT_PATH)

    i = args.split_idx

    # write the statistics into the file
    file_obj = open(args.output_file, "w")

    start_time = time.time()
    hist_dict, best_epoch, best_val_nll, _ = main(args, i)
    file_obj.write(f"train time for one split is: {(time.time() - start_time):.2f} seconds\n")
    file_obj.write('\n')
    
    #train_best_epoch = hist_dict["mae"]['train_loss'][best_epoch]
    #val_best_epoch = best_val_nll
    val_best_epoch = hist_dict["mae"]['valid_loss'][best_epoch]
    test_best_epoch = hist_dict["mae"]['test_loss'][best_epoch]
    
    # MICRO MAE Loss
    file_obj.write('\n')
    file_obj.write('MICRO MAE Best Epoch Statistics\n')
    file_obj.write("-"*50 + "\n")
    #file_obj.write(f"train | {train_best_epoch:.2f} \n")
    file_obj.write(f"valid | {val_best_epoch:.2f} \n")
    file_obj.write(f"test  | {test_best_epoch:.2f}\n")
    file_obj.write(f"best_epoch | {best_epoch}\n")
    
    # MACRO MAE loss
    file_obj.write("\n")
    valid_best_macro_mae = hist_dict["macro_mae"]["valid_loss"][best_epoch]
    test_best_macro_mae = hist_dict["macro_mae"]["test_loss"][best_epoch]
    file_obj.write(f"MACRO MAE Best Epoch Statistics\n")
    file_obj.write("-" * 50 + "\n")
    file_obj.write(f"valid | {valid_best_macro_mae:.2f}\n")
    file_obj.write(f"test  | {test_best_macro_mae:.2f}\n")
    
    # MICRO MSE and RMSE loss
    # statistics for MSE and RMSE
    valid_best_mse = hist_dict["mse"]["valid_loss"][best_epoch]
    test_best_mse = hist_dict["mse"]["test_loss"][best_epoch]
    file_obj.write("\n")
    file_obj.write(f"MICRO MSE Best Epoch Statistics\n")
    file_obj.write("-" * 50 + "\n")
    file_obj.write(f"valid | {valid_best_mse:.2f}\n")
    file_obj.write(f"test  | {test_best_mse:.2f}\n")
    
    file_obj.write('\n')
    file_obj.write(f"MICRO RMSE Best Epoch Statistics\n")
    file_obj.write("-" * 50 + "\n")
    file_obj.write(f"valid | {np.sqrt(valid_best_mse):.2f}\n")
    file_obj.write(f"test  | {np.sqrt(test_best_mse):.2f}\n")
    
    # MACRO RMSE
    #for _, _, _, i in result_list:
    valid_best_macro_rmse = hist_dict["macro_rmse"]["valid_loss"][best_epoch]
    test_best_macro_rmse = hist_dict["macro_rmse"]["test_loss"][best_epoch]
    file_obj.write("\n")
    file_obj.write(f"MACRO RMSE Best Epoch Statistics\n")
    file_obj.write("-" * 50 + "\n")
    file_obj.write(f"valid | {np.mean(valid_best_macro_rmse):.2f}\n")
    file_obj.write(f"test  | {np.mean(test_best_macro_rmse):.2f}\n")
    
    file_obj.close()


    ### plot the mean and std of the MAE loss over 10 splits ###
    # draw the plots
    w = 2.0
    epoch = np.arange(args.epochs) + 1

    fig, axs = plt.subplots(1, 3, figsize=(22, 5))
    
    axs[0].plot(epoch, hist_dict["mae"]["valid_loss"], c='darkblue', lw=w, label='valid_loss')
    axs[0].plot(epoch, hist_dict["mae"]["test_loss"], c='darkred', lw=w, label='test_loss')
    axs[1].plot(epoch, hist_dict["train_loss"]["generator"], c='black', lw=w, label='train_loss_generator')
    axs[2].plot(epoch, hist_dict["train_loss"]["discriminator"], c='black', lw=w, label='train_loss_discriminator')
        
    axs[0].set_ylabel("MAE Loss on Step Counts")
    axs[1].set_ylabel("Train Loss Generator")
    axs[2].set_ylabel("Train Loss Discriminator")
    
    axs[0].set_xlabel("Epochs")
    axs[1].set_xlabel("Epochs")
    axs[2].set_xlabel("Epochs")
    
    axs[0].legend()
    axs[1].legend()
    axs[2].legend()

    axs[0].grid(linestyle='--')
    axs[1].grid(linestyle='--')
    axs[2].grid(linestyle="--")

    plt.tight_layout()
    timestamp = datetime.now().strftime('%m%d.%H%M%S')
    plt.savefig(f"{OUT_PATH}/allofus_{timestamp}_usgan_{args.lr}_{args.batch_size}_{args.gene_rnn_dim}_{args.disc_rnn_dim}_{args.seed}_split_{args.split_idx}.png",\
                dpi=300)
    
    # copy the model from the VM disk to google cloud bucket
    if args.save:
        model_file_name = f"{OUT_PATH}/best_model_usgan_split_{args.split_idx}_seed_{args.seed}_gene_{args.gene_rnn_dim}_disc_{args.disc_rnn_dim}"
        if not args.pad_full_weeks:
            model_file_name += "_same_day"
        model_file_name += ".pth"
        os.system(f"gsutil -m cp {model_file_name} {os.getenv('WORKSPACE_BUCKET')+'/data/'}")
    
    if args.save:
        loss_file_name = f"{OUT_PATH}/loss_history_usgan_split_{args.split_idx}_seed_{args.seed}_gene_{args.gene_rnn_dim}_disc_{args.disc_rnn_dim}"
        if not args.pad_full_weeks:
            loss_file_name += "_same_day"
        loss_file_name += ".pkl"
        os.system(f"gsutil -m cp {loss_file_name} {os.getenv('WORKSPACE_BUCKET')+'/data/'}")
    
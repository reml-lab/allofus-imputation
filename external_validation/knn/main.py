"""
This script is to use KNN regressor to impute the missing values from the nearest neighbors retrieved by KNN 
on the external validation set
"""
import os
import pickle 
import argparse
import time
from tqdm import tqdm
import numpy as np
import pandas as pd
from scipy.special import softmax

import sys
sys.path.append("../..")

from utils.data_utils import pull_file, FILE_CACHE
from external_validation.extvalid_utils import get_hourly_data
from external_validation.extvalid_utils import HIGH_MISS_START_END_FILE, HIGH_MISS_SHIFT_FILE, LOW_MISS_START_END_FILE, LOW_MISS_SHIFT_FILE
from utils.train_utils import new_dir

def get_args():
    """
    parser the arguments to tune the models
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight', type=str, default="uniform")
    parser.add_argument('--rbf-param', type=float, default=1.0)
    parser.add_argument('--num-nn', type=int, default=15)
    parser.add_argument('--if-high-miss', action="store_true")
    args = parser.parse_args()

    return args

# mae 
def mae_loss(output, true, mask=None, norm=True):
    loss = np.abs(output - true)
    if mask is not None:
        #loss = loss * mask
        loss[mask==0.0] = 0  # in case there is nan in loss 
        if norm:
            if mask.sum() == 0:
                return loss.sum()
            else:
                return loss.sum() / mask.sum()
        else:
            return loss
    else:
        if norm:
            return loss.mean()
        else:
            return loss.sum()
        
# mse 
def mse_loss(output, true, mask=None, norm=True):
    loss = (output - true) ** 2
    if mask is not None:
        #loss = loss * mask
        loss[mask==0.0] = 0  # in case there is nan in loss 
        if norm:
            if mask.sum() == 0:
                return loss.sum()
            else:
                return loss.sum() / mask.sum()
        else:
            return loss
    else:
        if norm:
            return loss.mean()
        else:
            return loss.sum()
        
        
def lower_upper_bound_func(tensor, lower_bound, upper_bound):
    """
    The function to limit values in the tensor to be between lower_bound and upper_bound
    """
    
    tensor[tensor < lower_bound] = 0.0
    tensor[tensor > 1.5 * upper_bound] = 1.5 * upper_bound
    
    return tensor

class KNN_regressor:
    def __init__(self, weight='uniform', rbf_param=1.0, step_rate_mean=None, step_rate_std=None, max_step_rate=None):
        
        assert weight in ["uniform", "softmax"], "weight must be uniform or softmax!"
        assert rbf_param > 0, "rbf parameter must be positive!"
        assert step_rate_mean is not None, "please provide step_rate_mean!"
        assert step_rate_std is not None, "please provide step_rate_std!"
        assert max_step_rate is not None, "please provide max_step_rate!"
        
        self.weight = weight  # how to put the weight to the distance of neighbors 
        self.rbf_b = rbf_param  # rbf kernel parameter
        self.step_rate_mean = step_rate_mean
        self.step_rate_std = step_rate_std
        self.max_step_rate = max_step_rate
    
    def predict(self, feat, dist):
        """
        Args: 
            - feat: feature matrix ([N, num_nn, 2]) including the normalized step rate and computation mask
            - dist: distance matrix ([N, num_nn]) including the distance between the query and each NNs
        Outputs:
            - pred: predicted unnormalized step rate, shape: [N, ]
        """
        # get the mask
        nsr_mtx = feat[:, :, 0]
        mask_mtx = feat[:, :, 1]
        
        if self.weight == "uniform":
            pred = (nsr_mtx * mask_mtx).sum(axis=1) / (mask_mtx == 1).sum(axis=1)
            
        elif self.weight == "softmax":
            # dist = - self.rbf_b * (dist ** 2)
            dist = -self.rbf_b * dist  # note that the dist returned by FAISS is squared distance
            dist[mask_mtx == 0] = -1e9
            p_attn = softmax(dist, axis=1)
            pred = (p_attn * nsr_mtx).sum(axis=1)
        
        pred = pred * self.step_rate_std + self.step_rate_mean
        pred = lower_upper_bound_func(pred, 0.0, self.max_step_rate)
             
        return pred # [N, ]
    

def main(weight, rbf_param, ctx_len, num_nn, n_neighbors, file_obj, if_high_miss=True):
    """
    Compute both the Micro and Macro MAE for all participants in the cohort
    """
    # get the list of participant id in the cohort
    if if_high_miss:
        pull_file(HIGH_MISS_START_END_FILE)  
        pull_file(HIGH_MISS_SHIFT_FILE)
        df_cohort = pd.read_parquet(f"{FILE_CACHE}/{HIGH_MISS_START_END_FILE}")
        df_shift = pd.read_parquet(f"{FILE_CACHE}/{HIGH_MISS_SHIFT_FILE}")
    else:
        pull_file(LOW_MISS_START_END_FILE) 
        pull_file(LOW_MISS_SHIFT_FILE)
        df_cohort = pd.read_parquet(f"{FILE_CACHE}/{LOW_MISS_START_END_FILE}")
        df_shift = pd.read_parquet(f"{FILE_CACHE}/{LOW_MISS_SHIFT_FILE}")

    pid_list = df_cohort.index.tolist()
    
    # pull the knn raw feature for all participants
    if if_high_miss:
        foldername = f"knn_feat_extvalid_high_miss/"
    else:
        foldername = f"knn_feat_extvalid_low_miss/"
    if not os.path.exists(os.path.join(FILE_CACHE, foldername)):
        os.system(f"gsutil -m cp -r {os.getenv('WORKSPACE_BUCKET')}/data/{foldername} {FILE_CACHE}")  

    # lists to store the results for all splits and all participants
    test_mae_list = []
    test_mse_list = []
    test_num_list = []

    for pid in tqdm(pid_list):
        # get features and distance matrices from the participant
        if if_high_miss:
            filename = f"{FILE_CACHE}/knn_feat_extvalid_high_miss/knn_feat_{pid}_extvalid.pkl"
        else:
            filename = f"{FILE_CACHE}/knn_feat_extvalid_low_miss/knn_feat_{pid}_extvalid.pkl"
        with open(filename, "rb") as fin:
            knn_feat_pid_list = pickle.load(fin)
        assert len(knn_feat_pid_list)==1, f"{pid} has incorrect split data stored in the list!"

        # get step_rate_mean, step_rate_std for the participant
        _, step_rate_mean, step_rate_std = get_hourly_data( pid, df_cohort, df_shift, start_hour=6, end_hour=22, conv_feat=False, return_time_dict=False)
        
        for split_idx, knn_feat_dict in enumerate(knn_feat_pid_list):
            nn_all_feat = knn_feat_dict["nn_all_feat"]
            dist_all_hourly = knn_feat_dict["dist_all_hourly"]
            # get first num_nn for both matrices
            nn_all_feat = nn_all_feat[:, :num_nn, :]  # [N, num_nn, n_channel], n_channel=14
            dist_all_hourly = dist_all_hourly[:, :num_nn]  # [N, num_nn]

            ### Create the input and output ###
            # split it into the input features and output groundtruth
            input_feat_test = nn_all_feat[:, :, [0,1,2,3,4,5]]  # [N, num_nn, 6]
            # remove the first one by setting the compute mask as zero
            input_feat_test[:,0,1] = 0
            # get the output groundtruth
            # 9: steps, 8: valid minutes
            # 6: test mask
            test_step_gt = nn_all_feat[:, 0, [9, 8, 6]]  # [N, 3]

            # get max_step_rate
            test_steps = test_step_gt[:, 0]
            test_valid_minutes = test_step_gt[:, 1]
            test_mask = test_step_gt[:, 2]
            max_step_rate = (test_steps[test_mask==1] / test_valid_minutes[test_mask==1]).max()

            # define the model
            model = KNN_regressor(weight, rbf_param, step_rate_mean, step_rate_std, max_step_rate)

            # make the prediction
            #### test ####
            pred = model.predict(input_feat_test.copy(), dist_all_hourly.copy())
            pred = pred * test_step_gt[:, 1]
            test_loss = mae_loss(pred, test_step_gt[:, 0], test_step_gt[:, 2], norm=True)
            test_mse_loss = mse_loss(pred, test_step_gt[:, 0], test_step_gt[:, 2], norm=True)
            test_mae_list.append(test_loss)
            test_mse_list.append(test_mse_loss)
            test_num_list.append(test_step_gt[:, 2].sum())
            
            # # store the testing results
            knn_test_result_bundle = {"pred":pred[test_step_gt[:, 2]==1],
                                      "gt": test_step_gt[:, 0][test_step_gt[:, 2]==1],
                                      "test_mask": test_step_gt[:, 2]}
            if if_high_miss:
                with open(f"{FILE_CACHE}/knn_{weight}_pred_results_extvalid_high_miss/knn_pred_dict_pid_{pid}_split_{split_idx}.pkl", "wb") as fout:
                    pickle.dump(knn_test_result_bundle, fout)
            else:
                with open(f"{FILE_CACHE}/knn_{weight}_pred_results_extvalid_low_miss/knn_pred_dict_pid_{pid}_split_{split_idx}.pkl", "wb") as fout:
                    pickle.dump(knn_test_result_bundle, fout)

    # compute Micro and Macro MAE and RMSE across all participants
    test_micro_mae_list = []
    test_micro_rmse_list = []
    test_macro_mae_list = []
    test_macro_rmse_list = []
       
    # test
    test_micro_mae_split = (np.array(test_mae_list) * np.array(test_num_list)).sum() / np.array(test_num_list).sum()
    test_micro_mse_split = (np.array(test_mse_list) * np.array(test_num_list)).sum() / np.array(test_num_list).sum()
    test_micro_rmse_split = np.sqrt(test_micro_mse_split)
    test_macro_mae_split = np.mean(test_mae_list)
    test_macro_rmse_split = np.mean(np.sqrt(test_mse_list))

    # test
    test_micro_mae_list.append(test_micro_mae_split.item())  # each element is the micro mae for each split
    test_micro_rmse_list.append(test_micro_rmse_split.item())
    test_macro_mae_list.append(test_macro_mae_split.item())  # each element is the macro mae for each split
    test_macro_rmse_list.append(test_macro_rmse_split.item())

    # write the performance into the file
    #### MICRO MAE ####
    file_obj.write('MICRO MAE Statistics\n')
    file_obj.write("-"*50 + "\n")
    file_obj.write(f"test  | mean: {np.mean(test_micro_mae_list):.2f} | std: {np.std(test_micro_mae_list):.2f}\n")
    file_obj.write('\n')

    #### MICRO RMSE ####
    file_obj.write('MICRO RMSE Statistics\n')
    file_obj.write("-"*50 + "\n")
    file_obj.write(f"test  | mean: {np.mean(test_micro_rmse_list):.2f} | std: {np.std(test_micro_rmse_list):.2f}\n")
    file_obj.write('\n')

    #### MACRO MAE ####
    file_obj.write('MACRO MAE Statistics\n')
    file_obj.write("-"*50 + "\n")
    file_obj.write(f"test  | mean: {np.mean(test_macro_mae_list):.2f} | std: {np.std(test_macro_mae_list):.2f}\n")
    file_obj.write('\n')
    
    #### MACRO MAE ####
    file_obj.write('MACRO RMSE Statistics\n')
    file_obj.write("-"*50 + "\n")
    file_obj.write(f"test  | mean: {np.mean(test_macro_rmse_list):.2f} | std: {np.std(test_macro_rmse_list):.2f}\n")
    file_obj.write('\n')


if __name__ == "__main__":
    # write the statistics into the file
    # get arguments
    args = get_args()

    weight = args.weight
    rbf_param = args.rbf_param
    num_nn = args.num_nn  # will add one when calling the function
    ctx_len = 72
    n_neighbors = 50

    # build folders to store predicted results
    if args.if_high_miss:
        new_dir(f"{FILE_CACHE}/knn_{weight}_pred_results_extvalid_high_miss")
    else:
        new_dir(f"{FILE_CACHE}/knn_{weight}_pred_results_extvalid_low_miss")
    
    # build folder to store the loss history and the best model
    new_dir(f"../results")
    #OUT_PATH = f"../results/{args.weight}_{args.rbf_param}_{args.ctx_len}_{args.num_nn}_{args.mask_weight}.txt"
    if args.if_high_miss:
        OUT_PATH = f"../results/knn_results_{weight}_{rbf_param}_{num_nn}_{ctx_len}_{n_neighbors}_extvalid_high_miss.txt"
    else:
        OUT_PATH = f"../results/knn_results_{weight}_{rbf_param}_{num_nn}_{ctx_len}_{n_neighbors}_extvalid_low_miss.txt"
    file_obj = open(OUT_PATH, "w")

    # store all the results
    start_time = time.time()
    main(weight, rbf_param, ctx_len, num_nn+1, n_neighbors, file_obj, args.if_high_miss)
    print(f"running time: {(time.time() - start_time):.2f} seconds")
    file_obj.close()
    






            




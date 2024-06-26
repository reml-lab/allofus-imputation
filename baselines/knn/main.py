"""
This script is to use KNN regressor to impute the missing values from the nearest neighbors retrieved by KNN 
"""
import pickle 
import argparse
import time
from tqdm import tqdm
import numpy as np
import pandas as pd
from scipy.special import softmax

import sys
sys.path.append("../..")

from utils.data_utils import pull_file, FILE_CACHE, get_hourly_data
from utils.train_utils import new_dir

def get_args():
    """
    parser the arguments to tune the models
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight', type=str, default="uniform")
    parser.add_argument('--rbf-param', type=float, default=1.0)
    # parser.add_argument('--ctx-len', type=int, default=12)  # the number of hours on one side of the current hourly block
    parser.add_argument('--num-nn', type=int, default=15)
    # parser.add_argument('--mask-weight', type=float, default=1.0)
    # parser.add_argument('--n-neighbors', type=int, default=200)
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
    

def main(weight, rbf_param, ctx_len, num_nn, n_neighbors, file_obj):
    """
    Compute both the Micro and Macro MAE for all participants in the cohort
    """
    # get the list of participant id in the cohort
    pull_file("df_cohort_top100.parquet")  # created by get_cohort_aaai.ipynb
    df_cohort = pd.read_parquet(f"{FILE_CACHE}/df_cohort_top100.parquet")
    pid_list = df_cohort.index.tolist()
    
    # pull the knn raw feature for all participants
    for pid in tqdm(pid_list):
        pull_file(f"knn_raw_feat_{pid}.pkl")

    # lists to store the results for all splits and all participants
    train_mae_list = [[] for _ in range(10)]
    train_mse_list = [[] for _ in range(10)]
    train_num_list = [[] for _ in range(10)]

    valid_mae_list = [[] for _ in range(10)]
    valid_mse_list = [[] for _ in range(10)]
    valid_num_list = [[] for _ in range(10)]

    test_mae_list = [[] for _ in range(10)]
    test_mse_list = [[] for _ in range(10)]
    test_num_list = [[] for _ in range(10)]

    for pid in tqdm(pid_list):
        # get features and distance matrices for 10 splits from the participant
        filename = f"{FILE_CACHE}/knn_raw_feat_{pid}.pkl"
        with open(filename, "rb") as fin:
            knn_feat_pid_list = pickle.load(fin)
        assert len(knn_feat_pid_list)==10, f"{pid} has less split data stored in the list!"

        # get step_rate_mean, step_rate_std for the participant
        _, _, step_rate_mean, step_rate_std = get_hourly_data(pid, num_split=10, ks=(9, 15), conv_feat=True)
        
        for split_idx, knn_feat_dict in enumerate(knn_feat_pid_list):
            nn_all_feat = knn_feat_dict["nn_all_feat"]
            dist_all_hourly = knn_feat_dict["dist_all_hourly"]
            # get first num_nn for both matrices
            nn_all_feat = nn_all_feat[:, :num_nn, :]  # [N, num_nn, n_channel], n_channel=14
            dist_all_hourly = dist_all_hourly[:, :num_nn]  # [N, num_nn]

            ### Create the input and output ###
            # split it into the input features and output groundtruth
            input_feat_train = nn_all_feat[:, :, [0,1,4,5,6,7]]  # [N, num_nn, 6]
            input_feat_valid = nn_all_feat[:, :, [0,2,4,5,6,7]]
            input_feat_test = nn_all_feat[:, :, [0,3,4,5,6,7]]
            # remove the first one by setting the compute mask as zero
            input_feat_train[:,0,1] = 0
            input_feat_valid[:,0,1] = 0
            input_feat_test[:,0,1] = 0
            # get the output groundtruth
            # 13: steps, 12: valid minutes
            # 8, 9, 10: train, valid and test mask
            train_step_gt = nn_all_feat[:, 0, [13, 12, 8]]  # [N, 3]
            valid_step_gt = nn_all_feat[:, 0, [13, 12, 9]]
            test_step_gt = nn_all_feat[:, 0, [13, 12, 10]]

            # get max_step_rate
            train_steps = train_step_gt[:, 0]
            train_valid_minutes = train_step_gt[:, 1]
            train_mask = train_step_gt[:, 2]
            max_step_rate = (train_steps[train_mask==1] / train_valid_minutes[train_mask==1]).max()

            # define the model
            model = KNN_regressor(weight, rbf_param, step_rate_mean, step_rate_std, max_step_rate)

            # make the prediction
            #### train ####
            pred = model.predict(input_feat_train.copy(), dist_all_hourly.copy())
            pred = pred * train_step_gt[:, 1]
            train_loss = mae_loss(pred, train_step_gt[:, 0], train_step_gt[:, 2], norm=True)
            train_mse_loss = mse_loss(pred, train_step_gt[:, 0], train_step_gt[:, 2], norm=True)
            train_mae_list[split_idx].append(train_loss)
            train_mse_list[split_idx].append(train_mse_loss)
            train_num_list[split_idx].append(train_step_gt[:, 2].sum())

            #### valid ####
            pred = model.predict(input_feat_valid.copy(), dist_all_hourly.copy())
            pred = pred * valid_step_gt[:, 1]
            valid_loss = mae_loss(pred, valid_step_gt[:, 0], valid_step_gt[:, 2], norm=True)
            valid_mse_loss = mse_loss(pred, valid_step_gt[:, 0], valid_step_gt[:, 2], norm=True)
            valid_mae_list[split_idx].append(valid_loss)
            valid_mse_list[split_idx].append(valid_mse_loss)
            valid_num_list[split_idx].append(valid_step_gt[:, 2].sum())

            #### test ####
            pred = model.predict(input_feat_test.copy(), dist_all_hourly.copy())
            pred = pred * test_step_gt[:, 1]
            test_loss = mae_loss(pred, test_step_gt[:, 0], test_step_gt[:, 2], norm=True)
            test_mse_loss = mse_loss(pred, test_step_gt[:, 0], test_step_gt[:, 2], norm=True)
            test_mae_list[split_idx].append(test_loss)
            test_mse_list[split_idx].append(test_mse_loss)
            test_num_list[split_idx].append(test_step_gt[:, 2].sum())
            
            # # store the testing results
            # knn_test_result_bundle = {"pred":pred[test_step_gt[:, 2]==1],
            #                           "gt": test_step_gt[:, 0][test_step_gt[:, 2]==1],
            #                           "test_mask": test_step_gt[:, 2]}
            # with open(f"{FILE_CACHE}/knn_pred_dict_pid_{pid}_split_{split_idx}.pkl", "wb") as fout:
            #     pickle.dump(knn_test_result_bundle, fout)

    # compute Micro and Macro MAE and RMSE across all participants
    train_micro_mae_list = []
    train_micro_rmse_list = []
    train_macro_mae_list = []
    train_macro_rmse_list = []
    valid_micro_mae_list = []
    valid_micro_rmse_list = []
    valid_macro_mae_list = []
    valid_macro_rmse_list = []
    test_micro_mae_list = []
    test_micro_rmse_list = []
    test_macro_mae_list = []
    test_macro_rmse_list = []

    for split_idx in range(10):
        # train
        train_micro_mae_split = (np.array(train_mae_list[split_idx]) * np.array(train_num_list[split_idx])).sum() / np.array(train_num_list[split_idx]).sum()
        train_micro_mse_split = (np.array(train_mse_list[split_idx]) * np.array(train_num_list[split_idx])).sum() / np.array(train_num_list[split_idx]).sum()
        train_micro_rmse_split = np.sqrt(train_micro_mse_split)
        train_macro_mae_split = np.mean(train_mae_list[split_idx])
        train_macro_rmse_split = np.mean(np.sqrt(train_mse_list[split_idx]))
        # valid
        valid_micro_mae_split = (np.array(valid_mae_list[split_idx]) * np.array(valid_num_list[split_idx])).sum() / np.array(valid_num_list[split_idx]).sum()
        valid_micro_mse_split = (np.array(valid_mse_list[split_idx]) * np.array(valid_num_list[split_idx])).sum() / np.array(valid_num_list[split_idx]).sum()
        valid_micro_rmse_split = np.sqrt(valid_micro_mse_split)
        valid_macro_mae_split = np.mean(valid_mae_list[split_idx])
        valid_macro_rmse_split = np.mean(np.sqrt(valid_mse_list[split_idx]))
        # test
        test_micro_mae_split = (np.array(test_mae_list[split_idx]) * np.array(test_num_list[split_idx])).sum() / np.array(test_num_list[split_idx]).sum()
        test_micro_mse_split = (np.array(test_mse_list[split_idx]) * np.array(test_num_list[split_idx])).sum() / np.array(test_num_list[split_idx]).sum()
        test_micro_rmse_split = np.sqrt(test_micro_mse_split)
        test_macro_mae_split = np.mean(test_mae_list[split_idx])
        test_macro_rmse_split = np.mean(np.sqrt(test_mse_list[split_idx]))

        # train
        train_micro_mae_list.append(train_micro_mae_split.item())  # each element is the micro mae for each split
        train_micro_rmse_list.append(train_micro_rmse_split.item())
        train_macro_mae_list.append(train_macro_mae_split.item())  # each element is the macro mae for each split
        train_macro_rmse_list.append(train_macro_rmse_split.item())
        # valid
        valid_micro_mae_list.append(valid_micro_mae_split.item())  # each element is the micro mae for each split
        valid_micro_rmse_list.append(valid_micro_rmse_split.item())
        valid_macro_mae_list.append(valid_macro_mae_split.item())  # each element is the macro mae for each split
        valid_macro_rmse_list.append(valid_macro_rmse_split.item())
        # test
        test_micro_mae_list.append(test_micro_mae_split.item())  # each element is the micro mae for each split
        test_micro_rmse_list.append(test_micro_rmse_split.item())
        test_macro_mae_list.append(test_macro_mae_split.item())  # each element is the macro mae for each split
        test_macro_rmse_list.append(test_macro_rmse_split.item())

    # write the performance into the file
    #### MICRO MAE ####
    for i in range(10):
        file_obj.write(f"split {i} | train_micro_mae: {train_micro_mae_list[i]:.2f}| valid_micro_mae: {valid_micro_mae_list[i]:.2f} | test_micro_mae: {test_micro_mae_list[i]:.2f}\n")
    file_obj.write('\n')
    file_obj.write('MICRO MAE Statistics\n')
    file_obj.write("-"*50 + "\n")
    file_obj.write(f"train | mean: {np.mean(train_micro_mae_list):.2f} | std: {np.std(train_micro_mae_list):.2f}\n")
    file_obj.write(f"valid | mean: {np.mean(valid_micro_mae_list):.2f} | std: {np.std(valid_micro_mae_list):.2f}\n")
    file_obj.write(f"test  | mean: {np.mean(test_micro_mae_list):.2f} | std: {np.std(test_micro_mae_list):.2f}\n")
    file_obj.write(f"diff of valid and test mean: {np.abs(np.mean(valid_micro_mae_list)-np.mean(test_micro_mae_list)):.2f}\n")
    file_obj.write('\n')

    #### MICRO RMSE ####
    for i in range(10):
        file_obj.write(f"split {i} | train_micro_rmse: {train_micro_rmse_list[i]:.2f}| valid_micro_rmse: {valid_micro_rmse_list[i]:.2f} | test_micro_rmse: {test_micro_rmse_list[i]:.2f}\n")
    file_obj.write('\n')
    file_obj.write('MICRO RMSE Statistics\n')
    file_obj.write("-"*50 + "\n")
    file_obj.write(f"train | mean: {np.mean(train_micro_rmse_list):.2f} | std: {np.std(train_micro_rmse_list):.2f}\n")
    file_obj.write(f"valid | mean: {np.mean(valid_micro_rmse_list):.2f} | std: {np.std(valid_micro_rmse_list):.2f}\n")
    file_obj.write(f"test  | mean: {np.mean(test_micro_rmse_list):.2f} | std: {np.std(test_micro_rmse_list):.2f}\n")
    file_obj.write(f"diff of valid and test mean: {np.abs(np.mean(valid_micro_rmse_list)-np.mean(test_micro_rmse_list)):.2f}\n")
    file_obj.write('\n')

    #### MACRO MAE ####
    for i in range(10):
        file_obj.write(f"split {i} | train_macro_mae: {train_macro_mae_list[i]:.2f}| valid_macro_mae: {valid_macro_mae_list[i]:.2f} | test_macro_mae: {test_macro_mae_list[i]:.2f}\n")
    file_obj.write('\n')
    file_obj.write('MACRO MAE Statistics\n')
    file_obj.write("-"*50 + "\n")
    file_obj.write(f"train | mean: {np.mean(train_macro_mae_list):.2f} | std: {np.std(train_macro_mae_list):.2f}\n")
    file_obj.write(f"valid | mean: {np.mean(valid_macro_mae_list):.2f} | std: {np.std(valid_macro_mae_list):.2f}\n")
    file_obj.write(f"test  | mean: {np.mean(test_macro_mae_list):.2f} | std: {np.std(test_macro_mae_list):.2f}\n")
    file_obj.write(f"diff of valid and test mean: {np.abs(np.mean(valid_macro_mae_list)-np.mean(test_macro_mae_list)):.2f}\n")

    #### MACRO MAE ####
    for i in range(10):
        file_obj.write(f"split {i} | train_macro_rmse: {train_macro_rmse_list[i]:.2f}| valid_macro_rmse: {valid_macro_rmse_list[i]:.2f} | test_macro_rmse: {test_macro_rmse_list[i]:.2f}\n")
    file_obj.write('\n')
    file_obj.write('MACRO RMSE Statistics\n')
    file_obj.write("-"*50 + "\n")
    file_obj.write(f"train | mean: {np.mean(train_macro_rmse_list):.2f} | std: {np.std(train_macro_rmse_list):.2f}\n")
    file_obj.write(f"valid | mean: {np.mean(valid_macro_rmse_list):.2f} | std: {np.std(valid_macro_rmse_list):.2f}\n")
    file_obj.write(f"test  | mean: {np.mean(test_macro_rmse_list):.2f} | std: {np.std(test_macro_rmse_list):.2f}\n")
    file_obj.write(f"diff of valid and test mean: {np.abs(np.mean(valid_macro_rmse_list)-np.mean(test_macro_rmse_list)):.2f}\n")


if __name__ == "__main__":

    # get arguments
    args = get_args()

    weight = args.weight
    rbf_param = args.rbf_param
    num_nn = args.num_nn  # will add one when calling the function
    ctx_len = 72
    n_neighbors = 50

    # build folder to store the loss history and the best model
    new_dir(f"./results")
    OUT_PATH = f"./results/knn_results_{weight}_{rbf_param}_{num_nn}_{ctx_len}_{n_neighbors}.txt"
    file_obj = open(OUT_PATH, "w")

    # store all the results
    start_time = time.time()
    main(weight, rbf_param, ctx_len, num_nn+1, n_neighbors, file_obj)
    print(f"running time: {(time.time() - start_time):.2f} seconds")
    file_obj.close()
    






            





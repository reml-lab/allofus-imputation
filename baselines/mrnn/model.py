import math

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class FCN_Regression(nn.Module):
    """
    We include day of the week and hour of the day features here
    """
    def __init__(self):
        super(FCN_Regression, self).__init__()
        self.U = nn.Parameter(torch.Tensor(2, 33+24)) # 33 is dim of all features (include dw + hd)
        self.V = nn.Parameter(torch.Tensor(2, 4)) # 4 is dim of concat of x̃ and m
        #self.W = nn.Parameter(torch.Tensor(2, 2)) # 2 is dim of nsr and nhr 
        self.W = nn.Parameter(torch.Tensor(1, 2))  # we only predict step rate, no heart rate
        #self.alpha = nn.Parameter(torch.Tensor(2))
        self.alpha = nn.Parameter(torch.Tensor(1))
        self.beta = nn.Parameter(torch.Tensor(2))  # bias beta
        
        # to set U's diagonal w.r.t nsr and nhr as zeros
        #m_u = torch.ones(2, 33)   
        #m_u[0, 0], m_u[1,1] = 0, 0
        #self.register_buffer('m_u', m_u)
        
        # to set W as diagonal
        #m_w = torch.eye(2, 2)
        #m_w = torch.tensor([])
        #self.register_buffer('m_w', m_w)
        
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.U.size(0))
        self.U.data.uniform_(-stdv, stdv)
        self.V.data.uniform_(-stdv, stdv)
        self.W.data.uniform_(-stdv, stdv)
        self.alpha.data.uniform_(-stdv, stdv)
        self.beta.data.uniform_(-stdv, stdv)

    def forward(self, nsr_nhr_t, RNN_estimation_nsr_nhr_t, dw_hd_t, lapr_t, m_t, lower_bound, upper_bound):
        # create z_t 
        z_t = torch.cat([RNN_estimation_nsr_nhr_t, m_t], dim=-1)  # [bs, 4]
        # create x_t
        x_t = torch.cat([nsr_nhr_t, dw_hd_t, lapr_t], dim=-1) # [bs, 33]
        # compute h_t
        #h_t = F.relu(F.linear(x_t, self.U * self.m_u) + F.linear(z_t, self.V) + self.beta) # [bs, 2]
        h_t = F.relu(F.linear(x_t, self.U) + F.linear(z_t, self.V) + self.beta) # [bs, 2]
        # compute x_hat_t
        #x_hat_t = F.linear(h_t, self.W * self.m_w) + self.alpha  # [bs, 2]
        x_hat_t = F.linear(h_t, self.W) + self.alpha  # [bs, 1]
        # limit the normalized step rate range
        #x_hat_t[:, 0] = torch.clamp(x_hat_t[:, 0], min=lower_bound.squeeze(1), max=upper_bound.squeeze(1))
        # we can limit the range on the step rate level instead here on the normalized step rate level
        # x_hat_t = torch.clamp(x_hat_t, min=lower_bound.squeeze(1), max=upper_bound.squeeze(1))
        
        return x_hat_t
    

class MRNN_LAPR(nn.Module):
    def __init__(self, rnn_hidden_size, device):
        super(MRNN_LAPR, self).__init__()
        # data settings
        self.rnn_hidden_size = rnn_hidden_size
        self.device = device
        
        # normalized step rate
        self.f_rnn_nsr = nn.GRUCell(3, self.rnn_hidden_size)
        self.b_rnn_nsr = nn.GRUCell(3, self.rnn_hidden_size)

        # normalized heart rate
        self.f_rnn_nhr = nn.GRUCell(3, self.rnn_hidden_size)
        self.b_rnn_nhr = nn.GRUCell(3, self.rnn_hidden_size)
        
        self.rnn_cells = {"forward": [self.f_rnn_nsr, self.f_rnn_nhr], "backward": [self.b_rnn_nsr, self.b_rnn_nhr]}
        self.concated_hidden_project_nsr = nn.Linear(self.rnn_hidden_size * 2, 1)
        self.concated_hidden_project_nhr = nn.Linear(self.rnn_hidden_size * 2, 1)
        self.fcn_regression = FCN_Regression()

        # lapr
        self.conv_lapr = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=49, stride=1, padding=24, bias=False)
        self.ln_lapr = nn.LayerNorm(normalized_shape=145)
        # shared for key, query and value 
        self.pool = nn.AvgPool1d(kernel_size=7, stride=6)
        self.relu = nn.ReLU()
    
   
    def gene_hidden_states(self, data, direction):
        values = data[direction]["nsr_nhr_values"].to(self.device)  # [bs, khkw, 2]
        masks = data[direction]["masks"].to(self.device) # [bs, khkw, 2]
        deltas = data[direction]["deltas"].to(self.device)  # [bs, khkw, 2]

        # split them for step rates and heart rates
        values_nsr = values[:, :, 0].unsqueeze(-1)
        values_nhr = values[:, :, 1].unsqueeze(-1)
        masks_nsr = masks[:, :, 0].unsqueeze(-1)
        masks_nhr = masks[:, :, 1].unsqueeze(-1)
        deltas_nsr = masks[:, :, 0].unsqueeze(-1)
        deltas_nhr = masks[:, :, 1].unsqueeze(-1)

        hidden_states_collector_nsr, hidden_states_collector_nhr = [], []
        hidden_state_nsr = torch.zeros(
            (values.size()[0], self.rnn_hidden_size), device=self.device 
        )
        hidden_state_nhr = torch.zeros(
            (values.size()[0], self.rnn_hidden_size), device=self.device
        )

        for t in range(values.shape[1]):
            # for step rate
            x_nsr = values_nsr[:, t, :]
            m_nsr = masks_nsr[:, t, :]
            d_nsr = deltas_nsr[:, t, :]
            inputs_nsr = torch.cat([x_nsr, m_nsr, d_nsr], dim=1)
            hidden_state_nsr = self.rnn_cells[direction][0](inputs_nsr, hidden_state_nsr)
            hidden_states_collector_nsr.append(hidden_state_nsr)

            # for heart rate
            x_nhr = values_nhr[:, t, :]
            m_nhr = masks_nhr[:, t, :]
            d_nhr = deltas_nhr[:, t, :]
            inputs_nhr = torch.cat([x_nhr, m_nhr, d_nhr], dim=1)
            hidden_state_nhr = self.rnn_cells[direction][1](inputs_nhr, hidden_state_nhr)
            hidden_states_collector_nhr.append(hidden_state_nhr)

        return hidden_states_collector_nsr, hidden_states_collector_nhr
    
    
    def forward(self, data):
        
        hidden_states_f_nsr, hidden_states_f_nhr = self.gene_hidden_states(data, "forward")
        hidden_states_b_nsr, hidden_states_b_nhr = self.gene_hidden_states(data, "backward")
        hidden_states_b_nsr, hidden_states_b_nhr = hidden_states_b_nsr[::-1], hidden_states_b_nhr[::-1]
        
        values_nsr_nhr = data["forward"]["nsr_nhr_values"].to(self.device)
        masks = data["forward"]["masks"].to(self.device)
        values_dw_hd = data["forward"]["dw_hd_values"].to(self.device)
        values_lapr = data["forward"]["lapr_values"].to(self.device)
        
        max_sr = data["max_sr"].to(self.device)
        min_sr = torch.zeros_like(max_sr, device=self.device)
        sr_mean = data["sr_mean"].to(self.device)
        sr_std = data["sr_std"].to(self.device)
        upper_bound = (1.5 * max_sr - sr_mean) / sr_std
        lower_bound = (min_sr - sr_mean) / sr_std

        # here, we directly use the center hourly block
        # we don't need to consider other hourly blocks
        i = values_nsr_nhr.shape[1] // 2  
        
        # get all features and all masks
        x_nsr_nhr = values_nsr_nhr[:, i, :]
        m = masks[:, i, :]
        x_dw_hd = values_dw_hd[:, i, :]
        x_lapr = values_lapr[:, i, :].unsqueeze(1)

        # we need to process lapr feature through the lapr encoder
        bs, khkw, nsr_len = x_lapr.shape
        x_lapr = x_lapr.reshape(-1, nsr_len)
        # add the channel dimension
        x_lapr = x_lapr.unsqueeze(1)  # [bs*1, 1, 145]
        x_lapr = self.ln_lapr(self.conv_lapr(x_lapr)) # [bs*1, 1, 24]
        x_lapr = self.relu(x_lapr)
        x_lapr = self.pool(x_lapr)
        # reshape it back
        x_lapr = x_lapr.squeeze(1).reshape(bs, khkw, -1) # [33827, 135, 24]
        x_lapr = x_lapr.squeeze(1)
        
        # get hidden states for nsr and nhr
        h_f_nsr = hidden_states_f_nsr[i]
        h_b_nsr = hidden_states_b_nsr[i]

        h_f_nhr = hidden_states_f_nhr[i]
        h_b_nhr = hidden_states_b_nhr[i]

        h_nsr = torch.cat([h_f_nsr, h_b_nsr], dim=1)
        h_nhr = torch.cat([h_f_nhr, h_b_nhr], dim=1)
        
        # get the imputed data from RNN for each feature
        RNN_estimation_nsr = self.concated_hidden_project_nsr(h_nsr)  # x̃_t_nsr
        RNN_estimation_nhr = self.concated_hidden_project_nhr(h_nhr)  # x̃_t_nhr
        # Note that in the original paper, there is no m*x + (1-m)*RNN_estimation
        # therefore, we remove this process here
        # Note that we need to limit the range of the nsr
        RNN_estimation_nsr = torch.clamp(RNN_estimation_nsr, min=lower_bound, max=upper_bound)
        
        FCN_estimation_nsr = self.fcn_regression(
                                x_nsr_nhr, 
                                torch.cat([RNN_estimation_nsr, RNN_estimation_nhr], dim=-1),
                                x_dw_hd,
                                x_lapr,
                                m,
                                lower_bound,
                                upper_bound
                        )  # FCN estimation is output extimation
        
        
        # FCN_estimation_nsr = FCN_estimation[:, 0].unsqueeze(1)
        
        # convert it back to the unnormalized step rate
        FCN_estimation_sr = FCN_estimation_nsr * sr_std + sr_mean
        # limit the prediction to be the range of 0.0 to 1.5 * max_step_rate
        FCN_estimation_sr = torch.clamp(FCN_estimation_sr, min=torch.zeros_like(max_sr, device=self.device), max=1.5 * max_sr)
        
        return FCN_estimation_sr
    

def make_forward_backward_data_with_lapr(input_feat, lapr_feat, max_sr, sr_mean, sr_std, ks):
    # make the forward and backward input features for BRITS model
    # input_feat: [bs, 6, kh*kw]
    # lapr_feat: [bs, kh*kw, 145]
    # ["step_rate_norm", "mask_comp", "Day of Week", "Hour of Day", "time_axis", "heart_rate_norm"]
    
    data = {"forward":{}, "backward":{}}  # store all the data including both forward and backward
    
    ### step 1: reorder the feature ###
    input_feat = input_feat.reshape(input_feat.shape[0], input_feat.shape[1], ks[0], -1)  # row major
    input_feat = input_feat.transpose(-1, -2)  # column major
    input_feat = input_feat.reshape(input_feat.shape[0], input_feat.shape[1], -1)  # reorder into time order
    input_feat = input_feat.transpose(-1, -2) # put the time dimension on the second dim

    ### step 1.5: reorder the lapr ###
    lapr_feat = lapr_feat.reshape(lapr_feat.shape[0], ks[0], -1, lapr_feat.shape[-1])
    lapr_feat = lapr_feat.transpose(1, 2) # column major
    lapr_feat = lapr_feat.reshape(lapr_feat.shape[0], -1, lapr_feat.shape[-1]) # reorder into time order
    # time dimension has already been at the second dim
    
    ### step 2: split the features ###
    # normalized step rate, normalized heart rate, computational mask
    nsr_feat = input_feat[:, :, 0].unsqueeze(-1)
    nhr_feat = input_feat[:, :, 5].unsqueeze(-1)
    comp_mask = input_feat[:, :, 1].unsqueeze(-1)
    ###### MOST IMPORTANT!!!!!! ######
    comp_mask[:, comp_mask.shape[1]//2, :] = 0 # don't count the to-be-predicted point
    nsr_feat[comp_mask==0] = 0
    nhr_feat[comp_mask==0] = 0
    # check the correctness for the center hourly block
    assert nsr_feat[:, nsr_feat.shape[1]//2, :].sum().item()==0, "nsr_feat center hour is not masked!"
    assert nhr_feat[:, nhr_feat.shape[1]//2, :].sum().item()==0, "nhr_feat center hour is not masked!"
    # concatenate nsr and nhr
    nsr_nhr_values = torch.concat([nsr_feat, nhr_feat], dim=-1)
    
    # day of the week and hour of the day indicator
    dw_feat = input_feat[:, :, 2]
    hd_feat = input_feat[:, :, 3]
    # one hot encoding for these features
    dw_onehot = F.one_hot(dw_feat.long(), num_classes=7)
    hd_onehot = F.one_hot(hd_feat.long(), num_classes=24)
    # concatenate them together
    dw_hd_values = torch.cat([dw_onehot, hd_onehot], dim=-1)  # [bs, kh*kw, 31]
    
    ### step 3: get the deltas ###
    tp_values = input_feat[:, :, 4]  # get the time axis
    ## get the forward deltas ##
    deltas_forward = torch.zeros_like(tp_values)
    for h in range(1, tp_values.shape[1]):
        deltas_forward[:, h] = (tp_values[:, h] - tp_values[:, h-1]) + (1-comp_mask[:, h-1, :].squeeze(-1)) * deltas_forward[:, h-1]
    ## get the backward deltas ##
    # we then need reverse the mask_test to get the backed mask
    reverse_index = (comp_mask.shape[1]-1) - np.arange(comp_mask.shape[1])
    comp_mask_backward = comp_mask[:, reverse_index, :]
    # s[t-1]-s[t] but with backward order
    tp_values_backward = torch.abs((tp_values - tp_values[:, -1].unsqueeze(-1))[:,reverse_index])
    # get the backward deltas #
    deltas_backward = torch.zeros_like(tp_values_backward)
    for h in range(1, tp_values_backward.shape[1]):
        deltas_backward[:, h] = (tp_values_backward[:, h] - tp_values_backward[:, h-1]) + (1-comp_mask_backward[:, h-1].squeeze(-1)) * deltas_backward[:, h-1]
    
    ### step 3: put all the data into the dictionary ###
    data["forward"]["nsr_nhr_values"] = nsr_nhr_values  # [bs, khkw, 2]
    data["forward"]["dw_hd_values"] = dw_hd_values  # [bs, khkw, 31]
    data["forward"]["lapr_values"] = lapr_feat # [bs, khkw, 145]
    data["forward"]["masks"] = comp_mask.repeat(1, 1, 2)  # [bs, khkw, 1] note mask for heart rate and step rate are the same!
    # note we need to normalize the delta as well in order to make them into [0,1]
    # which is to ease the optimization
    data["forward"]["deltas"] = deltas_forward.unsqueeze(-1).repeat(1, 1, 2) / (24*ks[1])  # [bs, khkw, 2]
    
    data["backward"]["nsr_nhr_values"] = nsr_nhr_values[:, reverse_index, :]
    data["backward"]["dw_hd_values"] = dw_hd_values[:, reverse_index, :]
    data["backward"]["masks"] = comp_mask_backward.repeat(1, 1, 2)
    # note we need to normalize the delta as well in order to make them into [0,1]
    # which is to ease the optimization
    data["backward"]["deltas"] = deltas_backward.unsqueeze(-1).repeat(1, 1, 2) / (24*ks[1]) # [bs, khkw, 2]
    
    data["max_sr"] = max_sr
    data["sr_mean"] = sr_mean
    data["sr_std"] = sr_std
    
    return data

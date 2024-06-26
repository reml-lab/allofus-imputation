import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import math
import argparse


class FeatureRegression(nn.Module):
    def __init__(self, input_size=33):
        super(FeatureRegression, self).__init__()
        self.build(input_size)

    def build(self, input_size):
        ## Note that different from the original BRITS model, we concatenate variables which are always
        ## observed here (i.e. dayweek and hour of the day one-hot vectors) 
        ## so here input_size is the total size of step rate, heart rate and two one-hot vectors
        ## total size is 1+1+7+24 = 33
        ## however, we only regress for step rate and heart rate, therefore the output size of the linear
        ## layer is 2
        self.W = nn.Parameter(torch.Tensor(2, input_size))
        self.b = nn.Parameter(torch.Tensor(2))

        #m = torch.ones(input_size, input_size) - torch.eye(input_size, input_size)
        m = torch.ones(2, input_size)
        m[0, 0], m[1,1] = 0, 0
        self.register_buffer('m', m)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(0))
        self.W.data.uniform_(-stdv, stdv)
        if self.b is not None:
            self.b.data.uniform_(-stdv, stdv)

    def forward(self, x):
        z_h = F.linear(x, self.W * self.m, self.b)
        return z_h
    

class TemporalDecay(nn.Module):
    def __init__(self, input_size, output_size, diag = False):
        super(TemporalDecay, self).__init__()
        self.diag = diag

        self.build(input_size, output_size)

    def build(self, input_size, output_size):
        self.W = nn.Parameter(torch.Tensor(output_size, input_size))
        self.b = nn.Parameter(torch.Tensor(output_size))

        if self.diag == True:
            assert(input_size == output_size)
            m = torch.eye(input_size, input_size)
            self.register_buffer('m', m)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(0))
        self.W.data.uniform_(-stdv, stdv)
        if self.b is not None:
            self.b.data.uniform_(-stdv, stdv)

    def forward(self, d):
        if self.diag == True:
            gamma = F.relu(F.linear(d, self.W * self.m, self.b))
        else:
            gamma = F.relu(F.linear(d, self.W, self.b))
        gamma = torch.exp(-gamma)
        return gamma
    

class Model(nn.Module):
    def __init__(self, rnn_hid_size, device, no_hr_loss):
        super(Model, self).__init__()

        self.rnn_hid_size = rnn_hid_size
        self.device = device
        self.no_hr_loss = no_hr_loss  # we don't backprop heart rate loss

        self.build()

    def build(self):

        self.rnn_cell = nn.LSTMCell(2 * 2, self.rnn_hid_size)
        
        self.temp_decay_h = TemporalDecay(input_size=2, output_size=self.rnn_hid_size, diag=False)
        self.temp_decay_x = TemporalDecay(input_size=2, output_size=2, diag = True)

        ### define hist regression, and feature regression ###
        self.hist_reg = nn.Linear(self.rnn_hid_size, 2)
        self.feat_reg = FeatureRegression(33)  # steprate, heartrate, dayweek, hourofday
        self.weight_combine = nn.Linear(2 * 2, 2)
        
    def forward(self, data, direct):
        # Original sequence with 24 time steps
        nsr_nhr_values = data[direct]["nsr_nhr_values"].to(self.device)
        dw_hd_values = data[direct]["dw_hd_values"].to(self.device)
        masks = data[direct]["masks"].to(self.device)
        deltas = data[direct]["deltas"].to(self.device)
        max_sr = data["max_sr"].to(self.device)
        sr_mean = data["sr_mean"].to(self.device)
        sr_std = data["sr_std"].to(self.device)

        min_sr = torch.zeros_like(max_sr).to(self.device)
        upper_bound = (1.5 * max_sr - sr_mean) / sr_std
        lower_bound = (min_sr - sr_mean) / sr_std

        # Note we need to put them into the buffer using register buffer
        h = torch.zeros((nsr_nhr_values.size()[0], self.rnn_hid_size)).to(self.device)
        c = torch.zeros((nsr_nhr_values.size()[0], self.rnn_hid_size)).to(self.device)

        x_loss = 0.0
        num_m = 0.0  # number of positions the model is evaluated on (# of observed positions for all features)

        imputations = []

        for t in range(nsr_nhr_values.shape[1]):
            x = nsr_nhr_values[:, t, :]
            x_obs = dw_hd_values[:, t, :]
            m = masks[:, t, :]
            d = deltas[:, t, :]

            gamma_h = self.temp_decay_h(d)
            gamma_x = self.temp_decay_x(d)

            h = h * gamma_h

            x_h = self.hist_reg(h)
            x_h = torch.clamp(x_h, min=lower_bound, max=upper_bound)  # we need to limit the range of predicted values 
            if self.no_hr_loss:
                x_loss += torch.sum((torch.abs(x - x_h) * m)[:, 0]) / (torch.sum(m[:, 0]) + 1e-5)  # m.shape: [bs, 2]
            else:
                x_loss += torch.sum(torch.abs(x - x_h) * m) / (torch.sum(m) + 1e-5)
            
            x_c =  m * x +  (1 - m) * x_h
            ### concatenate x_c and x_obs(dayweek and hourofday) ###
            x_c = torch.concat([x_c, x_obs], dim=-1)

            z_h = self.feat_reg(x_c)
            z_h = torch.clamp(z_h, min=lower_bound, max=upper_bound)  # we need to limit the predicted value
            if self.no_hr_loss:
                x_loss += torch.sum((torch.abs(x - z_h) * m)[:, 0]) / (torch.sum(m[:, 0]) + 1e-5)
            else:
                x_loss += torch.sum(torch.abs(x - z_h) * m) / (torch.sum(m) + 1e-5)

            alpha = self.weight_combine(torch.cat([gamma_x, m], dim = 1))

            c_h = alpha * z_h + (1 - alpha) * x_h
            c_h = torch.clamp(c_h, min=lower_bound, max=upper_bound)  # we need to limit the predicted value
            if self.no_hr_loss:
                x_loss += torch.sum((torch.abs(x - c_h) * m)[:, 0]) / (torch.sum(m[:, 0]) + 1e-5)
            else:
                x_loss += torch.sum(torch.abs(x - c_h) * m) / (torch.sum(m) + 1e-5)

            c_c = m * x + (1 - m) * c_h

            inputs = torch.cat([c_c, m], dim = 1)

            h, c = self.rnn_cell(inputs, (h, c))

            imputations.append(c_c.unsqueeze(dim = 1))

        imputations = torch.cat(imputations, dim = 1)

        return {'loss': x_loss, 'imputations': imputations}

    def run_on_batch(self, data, optimizer, epoch = None):
        ret = self(data, direct = 'forward')

        if optimizer is not None:
            optimizer.zero_grad()
            ret['loss'].backward()
            optimizer.step()

        return ret

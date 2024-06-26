import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import math
import argparse

from baselines.brits import rits
from sklearn import metrics

class Model(nn.Module):
    def __init__(self, rnn_hid_size, device, no_hr_loss):
        super(Model, self).__init__()

        self.rnn_hid_size = rnn_hid_size
        self.device = device
        self.no_hr_loss = no_hr_loss

        self.build()

    def build(self):
        self.rits_f = rits.Model(self.rnn_hid_size, self.device, self.no_hr_loss)
        self.rits_b = rits.Model(self.rnn_hid_size, self.device, self.no_hr_loss)

    def forward(self, data):
        ret_f = self.rits_f(data, 'forward')
        ret_b = self.rits_b(data, 'backward')
        # ret_b = self.reverse(self.rits_b(data, 'backward'))

        ret = self.merge_ret(ret_f, ret_b)

        return ret

    def merge_ret(self, ret_f, ret_b):
        loss_f = ret_f['loss']
        loss_b = ret_b['loss']
        if self.no_hr_loss:
            loss_c = self.get_consistency_loss(ret_f['imputations'][:, :, 0], ret_b['imputations'][:, :, 0])
        else:
            loss_c = self.get_consistency_loss(ret_f['imputations'], ret_b['imputations'])

        loss = loss_f + loss_b + loss_c

        imputations = (ret_f['imputations'] + ret_b['imputations']) / 2

        ret_f['loss'] = loss
        ret_f['imputations'] = imputations

        return ret_f

    def get_consistency_loss(self, pred_f, pred_b):
        loss = torch.abs(pred_f - pred_b).mean() * 1e-1
        return loss

    # def reverse(self, ret):
    #     def reverse_tensor(tensor_):
    #         if tensor_.dim() <= 1:
    #             return tensor_
    #         indices = range(tensor_.size()[1])[::-1]
    #         indices = Variable(torch.LongTensor(indices), requires_grad = False)

    #         if torch.cuda.is_available():
    #             indices = indices.cuda()

    #         return tensor_.index_select(1, indices)

    #     for key in ret:
    #         ret[key] = reverse_tensor(ret[key])

    #     return ret

    def run_on_batch(self, data, optimizer, epoch=None):
        ret = self(data)

        if optimizer is not None:
            optimizer.zero_grad()
            ret['loss'].backward()
            optimizer.step()

        return ret


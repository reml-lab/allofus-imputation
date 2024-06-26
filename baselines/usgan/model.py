"""
USGAN generator and discriminator
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from baselines.brits import rits

# Modified based on BRITS 
class Generator(nn.Module):
    def __init__(self, rnn_hid_size, device):
        super(Generator, self).__init__()

        self.rnn_hid_size = rnn_hid_size
        self.device = device

        self.build()

    def build(self):
        self.rits_f = rits.Model(self.rnn_hid_size, self.device)
        self.rits_b = rits.Model(self.rnn_hid_size, self.device)

    def forward(self, data):
        ret_f = self.rits_f(data, 'forward')
        ret_b = self.reverse(self.rits_b(data, 'backward'))

        ret = self.merge_ret(ret_f, ret_b)

        return ret

    def merge_ret(self, ret_f, ret_b):
        loss_f = ret_f['loss']
        loss_b = ret_b['loss']
        loss_c = self.get_consistency_loss(ret_f['imputations'], ret_b['imputations'])

        loss = loss_f + loss_b + loss_c

        imputations = (ret_f['imputations'] + ret_b['imputations']) / 2

        ret_f['loss'] = loss
        ret_f['imputations'] = imputations

        return ret_f

    def get_consistency_loss(self, pred_f, pred_b):
        loss = torch.abs(pred_f - pred_b).mean() * 1e-1
        return loss

    def reverse(self, ret):
        def reverse_tensor(tensor_):
            if tensor_.dim() <= 1:
                return tensor_
            # tensor_ shape: [bs, seq_len, num_feat]
            indices = range(tensor_.size()[1])[::-1]
            indices = Variable(torch.LongTensor(indices), requires_grad = False)
            indices = indices.to(self.device)
            # if torch.cuda.is_available():
            #     indices = indices.cuda()
            return tensor_.index_select(1, indices)

        for key in ret:
            ret[key] = reverse_tensor(ret[key])

        return ret

    def run_on_batch(self, data, optimizer, epoch=None):
        ret = self(data)

        if optimizer is not None:
            optimizer.zero_grad()
            ret['loss'].backward()
            optimizer.step()

        return ret
    

# this is the modification of the usgan implementation from PyPOTS
# https://github.com/WenjieDu/PyPOTS/blob/main/pypots/imputation/usgan/modules/submodules.py
class Discriminator(nn.Module):
    def __init__(self, rnn_hid_size, hint_rate=0.8, device="cpu"):
        super(Discriminator, self).__init__()

        self.rnn_hid_size = rnn_hid_size
        self.hint_rate = hint_rate # 0.8 is default in the paper
        self.device = device

        self.build()

    def build(self):
        # note that the input dimension of biRNN is 3
        # which consists of generated normalized step rate, normalized heart rate
        # and the R matrix. Since the missing indicator are the same for heart rate and step rate
        # we only use one dimension, instead of redundent R matrix for this. 
        self.biRNN = nn.GRU(3, self.rnn_hid_size, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(2 * self.rnn_hid_size, 1) # * 2 is because RNN is bidirectional

    def forward(self, imputed_X, miss_mask):
        """
        imputed_X: shape [bs, kwkh, 2], imputed normalized step rates and heart rates from the generator
        miss_mask: shape [bs, kwkh, 2], 1 ==> non-miss, 0 ==> miss
        """
        # generate R matrix from miss_mask
        hint = (torch.rand((miss_mask.shape[0], miss_mask.shape[1], 1), dtype=torch.float, device=self.device) < self.hint_rate)
        hint = hint.int()
        h = hint * miss_mask[:, :, 0].unsqueeze(-1) + (1 - hint) * 0.5
        # get the probability
        x_in = torch.cat([imputed_X, h], dim=-1)
        out, _ = self.biRNN(x_in)  # [bs, kwkh, 2*rnn_hid_size]
        logit = self.linear(out) # [bs, kwkw, 1]

        return logit
    
    def run_on_batch(self, data, optimizer, epoch=None):
        ret = self(data)

        if optimizer is not None:
            optimizer.zero_grad()
            ret['loss'].backward()
            optimizer.step()

        return ret
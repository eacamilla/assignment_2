################################################################################
# MIT License
#
# Copyright (c) 2019
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course | Fall 2019
# Date Created: 2019-09-06
################################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

################################################################################

class LSTM(nn.Module):

    def __init__(self, seq_length, input_dim, num_hidden, num_classes,batch_size, device='cpu'):
        super(LSTM, self).__init__()
        # Initialization here ..
        self.num_hidden = num_hidden
        self.batch_size = batch_size
        self.seq_length = seq_length


        self.W_gx = nn.Parameter(torch.randn((num_hidden, input_dim)),requires_grad = True)
        self.W_gh = nn.Parameter(torch.randn((num_hidden, num_hidden)),requires_grad = True)
        self.bias_g = nn.Parameter(torch.zeros((num_hidden, 1)),requires_grad = True)

        self.W_ix = nn.Parameter(torch.randn((num_hidden, input_dim)),requires_grad = True)
        self.W_ih = nn.Parameter(torch.randn((num_hidden, num_hidden)),requires_grad = True)
        self.bias_i = nn.Parameter(torch.zeros((num_hidden, 1)),requires_grad = True)

        self.W_fx = nn.Parameter(torch.randn((num_hidden, input_dim)),requires_grad = True)
        self.W_fh = nn.Parameter(torch.randn((num_hidden, num_hidden)),requires_grad = True)
        self.bias_f = nn.Parameter(torch.zeros((num_hidden, 1)),requires_grad = True)

        self.W_ox = nn.Parameter(torch.randn((num_hidden, input_dim)),requires_grad = True)
        self.W_oh = nn.Parameter(torch.randn((num_hidden, num_hidden)),requires_grad = True)
        self.bias_o = nn.Parameter(torch.randn((num_hidden, 1)),requires_grad = True)


        self.W_ph = nn.Parameter(torch.randn((num_classes, num_hidden)),requires_grad = True)
        self.bias_p = nn.Parameter(torch.zeros((num_classes, 1)),requires_grad = True)



    def forward(self, x):

        h_tmin1 = torch.zeros(self.num_hidden, self.batch_size)
        c_tmin1 = torch.zeros(self.num_hidden, self.batch_size)

        for t in range(self.seq_length):
            g = torch.tanh(self.W_gx @ x[:,t].view(1,-1) + self.W_gh @ h_tmin1 + self.bias_g)
            i = nn.functional.sigmoid(self.W_ix @ x[:,t].view(1,-1) + self.W_ih @ h_tmin1 + self.bias_i)
            f = nn.functional.sigmoid(self.W_fx @ x[:, t].view(1, -1) + self.W_fh @ h_tmin1 + self.bias_f)
            o = nn.functional.sigmoid(self.W_ox @ x[:, t].view(1, -1) + self.W_oh @ h_tmin1 + self.bias_o)
            c = g * i + c_tmin1 * f
            h = torch.tanh(c) * o

            h_tmin1 = h
            c_tmin1 = c

        p = (self.W_ph @ h + self.bias_p).transpose(1,0)
        y = nn.functional.softmax(p)

        return y






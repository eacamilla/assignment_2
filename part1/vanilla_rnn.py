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
# model = VanillaRNN(config.input_length, config.input_dim, config.num_hidden,
#                    config.num_classes, config.batch_size, device)


class VanillaRNN(nn.Module):

    def __init__(self, seq_length, input_dim, num_hidden, num_classes,batch_size, device='cpu'):
        super(VanillaRNN, self).__init__()
        # Initialization here ...
        self.num_hidden = num_hidden
        self.batch_size = batch_size
        self.seq_length = seq_length

        #input-to-hidden
        self.W_hx = nn.Parameter(torch.randn((num_hidden, input_dim)),requires_grad = True)

        #hidden-to-hidden/recurrent
        self.W_hh = nn.Parameter(torch.randn((num_hidden, num_hidden)),requires_grad = True)

        #hidden-to-output
        self.W_ph = nn.Parameter(torch.randn((num_classes, num_hidden)),requires_grad = True)

        #Bias hidden
        self.bias_h = nn.Parameter(torch.zeros((num_hidden, 1)),requires_grad = True)

        #Bias output
        self.bias_p = nn.Parameter(torch.zeros((num_classes)),requires_grad = True)

    def forward(self, x):
        # Implementation here ...
        #For the forward pass you will need Python’s for-loop to step through time.
        h_tmin1 = torch.zeros(self.num_hidden, self.batch_size)

        for t in range(self.seq_length):
            h_t = torch.tanh(self.W_hx @ x[:,t].view(1,-1) + self.W_hh @ h_tmin1 + self.bias_h)
            print(h_t.shape)
            h_tmin1 = h_t

            p = h_t.t() @ self.W_ph.t() + self.bias_p


        return p

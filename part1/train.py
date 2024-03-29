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

import argparse
import time
from datetime import datetime
import numpy as np

import torch
from torch.utils.data import DataLoader

from part1.dataset import PalindromeDataset
from part1.vanilla_rnn import VanillaRNN
from part1.lstm import LSTM

# You may want to look into tensorboard for logging
# from torch.utils.tensorboard import SummaryWriter

################################################################################

def train(config):

    assert config.model_type in ('RNN', 'LSTM')

    # Initialize the device which to run the model on
    #device = torch.device(config.device)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize the model that we are going to use
    if config.model_type is 'RNN':
        model = VanillaRNN(seq_length = config.input_length, input_dim= config.input_dim, num_hidden = config.num_hidden,
                           num_classes = config.num_classes, batch_size = config.batch_size, device= device)
    if config.model_type is 'LSTM':
        model = LSTM(seq_length = config.input_length, input_dim= config.input_dim, num_hidden = config.num_hidden,
                           num_classes = config.num_classes, batch_size = config.batch_size, device= device)



    # Initialize the dataset and data loader (note the +1)
    dataset = PalindromeDataset(config.input_length+1)
    data_loader = DataLoader(dataset, config.batch_size, num_workers=1)

    # Setup the loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.RMSprop(model.parameters(), config.learning_rate)
    #optimizer = torch.optim.Adam(model.parameters(), config.learning_rate)

    Accuracy = []


    for step, (batch_inputs, batch_targets) in enumerate(data_loader):

        # Only for time measurement of step through network
        t1 = time.time()

        y = model.forward(batch_inputs.to(device))
        loss = criterion(y, batch_targets.to(device))
        loss.backward()

        ############################################################################
        # QUESTION: what happens here and why?
        # limits the size of the parameter updates by scaling the gradients down
        # Should be placed after loss.backward() but before optimizer.step()
        ############################################################################
        torch.nn.utils.clip_grad_norm(model.parameters(), max_norm=config.max_norm)
        ############################################################################

        optimizer.step()


        loss = loss.item()

        acc_in = np.argmax(y.cpu().detach().numpy(), axis=1) == batch_targets.cpu().detach().numpy()
        accuracy = np.sum(acc_in)/ batch_targets.shape[0]

        Accuracy.append(accuracy)

        # Just for time measurement
        t2 = time.time()
        examples_per_second = config.batch_size/float(t2-t1)

        if step % 10 == 0:

            print("[{}] Train Step {:04d}/{:04d}, Batch Size = {}, Examples/Sec = {:.2f}, "
                  "Accuracy = {:.2f}, Loss = {:.3f}".format(
                    datetime.now().strftime("%Y-%m-%d %H:%M"), step,
                    config.train_steps, config.batch_size, examples_per_second,
                    accuracy, loss
            ))

        if step == config.train_steps:
            # If you receive a PyTorch data-loader error, check this bug report:
            # https://github.com/pytorch/pytorch/pull/9655
            break

    print('Done training. :)')


 ################################################################################
 ################################################################################

if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--model_type', type=str, default="RNN", help="Model type, should be 'RNN' or 'LSTM'")
    #parser.add_argument('--model_type', type=str, default="LSTM", help="Model type, should be 'RNN' or 'LSTM'")


    parser.add_argument('--input_length', type=int, default=10, help='Length of an input sequence')
    #adjust input length for different palindrome lengths
    parser.add_argument('--input_dim', type=int, default=1, help='Dimensionality of input sequence')
    parser.add_argument('--num_classes', type=int, default=10, help='Dimensionality of output sequence')
    parser.add_argument('--num_hidden', type=int, default=128, help='Number of hidden units in the model')
    parser.add_argument('--batch_size', type=int, default=128, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--train_steps', type=int, default=10000, help='Number of training steps')
    parser.add_argument('--max_norm', type=float, default=10.0)
    parser.add_argument('--device', type=str, default="cuda:0", help="Training device 'cpu' or 'cuda:0'")

    config = parser.parse_args()


    #Loop over palindrome length and different seeds fixme
    # Train the model
    train(config)

    #Create plot fixme
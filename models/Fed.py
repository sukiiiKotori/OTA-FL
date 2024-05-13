#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torch import nn


def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg

#parameter 1 is a reference, so we modify it directly.
def FedAvg_Air(weight_global, grad, args):
    lr = args.lr
    data_sum = args.num_dataset
    for k in weight_global.keys():
        weight_global[k] -= lr * grad[k] / data_sum
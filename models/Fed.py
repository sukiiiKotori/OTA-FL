#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
import numpy as np
from torch import nn


def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg

#parameter weight_global is a reference, so we modify it directly.
def FedAvg_Air(weight_global, grad, total_size, args):
    lr = args.lr
    idx = 0
    for k in weight_global.keys():
        shape = np.array(weight_global[k].size())
        if len(shape):
            lenth = np.prod(shape)
            weight_global[k] -= torch.from_numpy(np.reshape(grad[idx:idx+lenth],shape)).float().to(args.device) * lr / total_size
            idx += lenth
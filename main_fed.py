#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch

from utils.sampling import mnist_iid, mnist_noniid, cifar_iid
from utils.options import args_parser
from utils.optimization import *
from models.Update import LocalUpdate
from models.Nets import MLP, CNNMnist, CNNCifar
from models.Fed import *
from models.test import test_img
from AirComp import *
from sensing import *


if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    # load dataset and split users
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
        # sample users
        if args.iid:
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            dict_users = mnist_noniid(dataset_train, args.num_users)
    elif args.dataset == 'cifar':
        trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=trans_cifar)
        if args.iid:
            dict_users = cifar_iid(dataset_train, args.num_users)
        else:
            exit('Error: only consider IID setting in CIFAR10')
    else:
        exit('Error: unrecognized dataset')
    img_size = dataset_train[0][0].shape

    # build model
    if args.model == 'cnn' and args.dataset == 'cifar':
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)
    else:
        exit('Error: unrecognized model')
    print(net_glob)
    net_glob.train()

    # copy weights
    w_glob = net_glob.state_dict()

    # training
    loss_train = []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    val_acc_list, net_list = [], []

    #initialize simulation settings
    D = 30000 #number of model weights
    v_max = 3.0
    u_max = 0.1
    dist_max = 1000.0
    noise_g = (np.random.randn(args.N,D)+1j*np.random.randn(args.N,D)) / 2**0.5
    noise_u = (np.random.randn(args.N, args.num_users)+1j*np.random.randn(args.N, args.num_users)) / 2**0.5
    noise_v = (np.random.randn(args.N, args.num_users)+1j*np.random.randn(args.N, args.num_users)) / 2**0.5
    num_dataset = []
    if args.fading:#consider channel fading
        # module (0,1)
        H_mod = np.random.uniform(low=0.0, high=1.0, size=(args.N, args.num_users))
        # phase (0,2π)
        H_phase = np.random.uniform(low=0.0, high=2*np.pi, size=(args.N, args.num_users))
        # H = (0,1)e^(j(0,2π))
        H = H_mod * np.exp(1j * H_phase)
        F = optimal_beamforming() #to be continue
    else:
        # not consider channel fading, so every term in f is equal to 1/sqrt(N)
        H = np.ones((args.N, args.num_users), dtype = complex)
        F = [1.0 / np.sqrt(args.N)] * args.N
    # derive optimal transmit power
    # not consider user selection M and beamforming vector f

    if args.all_clients:
        M = [i for i in range(args.num_users)]
        P_u, P_v, P_g = optimal_power()
    else:
        P_u, P_v, P_g, M = optimal_power_selection()
    
    for iter in range(args.epochs):
        loss_locals = []
        grad_locals = []
        mean_locals = []
        var_locals = []
        for idx in M:
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            grad, grad_mean, grad_var, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
            #train and collect results
            grad_locals.append(copy.deepcopy(grad))
            loss_locals.append(copy.deepcopy(loss))
            mean_locals.append(grad_mean)
            var_locals.append(grad_var)                                                                
        
        # transmit and receive variances/ sign of means
        # var_receive: list [var1, var2, ..., varm]; sign_mean: list [+1, -1, -1, +1]
        var_receive, sign_mean = transmission_var(var_locals, mean_locals, P_v, F, H, v_max, noise_v)

        # get transmit power of gradients and eta for AirComp
        p_gm, eta = get_args(var_receive, num_dataset, P_g, F, H)

        # transmit gradients with AirComp
        # signal_grad: dimension: m x D 
        signal_grad = transmit_grad(grad_locals, mean_locals, var_locals, p_gm)

        # apply matched filtering to get distance
        # to be continue
        dist_locals = matched_filtering(signal_grad)
        
        # recieve gradients with AirComp
        # at this time, means have not been add to the grads
        grad_receive = receive_grad(F, H, signal_grad, eta, noise_g)

        # transmit and receive means / quantilized angle to transfer distances
        # mean_abs_receive: list [mean_abs1, mean_abs2, ..., mean_absm]; angle dist: list [angle1, angle2, ..., anglem]
        # dist_max: to quantilize angle
        mean_abs_receive, dist_receive = transmission_mean(mean_locals, dist_locals, P_u, F, H, u_max, dist_max, noise_u)

        # formula (18): u = \sigma(Km * um)
        bias = np.array(num_dataset) @ (np.array(mean_abs_receive) * np.array(sign_mean))

        # add received means
        for k in grad_receive.keys():
            grad_receive[k] += bias

        #update global weights
        FedAvg_Air(w_glob, grad_receive, args)

        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)

        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
        loss_train.append(loss_avg)

    # plot loss curve
    plt.figure()
    plt.plot(range(len(loss_train)), loss_train)
    plt.ylabel('train_loss')
    plt.savefig('./save/fed_{}_{}_{}_iid{}.png'.format(args.dataset, args.model, args.epochs, args.iid))

    # testing
    net_glob.eval()
    acc_train, loss_train = test_img(net_glob, dataset_train, args)
    acc_test, loss_test = test_img(net_glob, dataset_test, args)
    print("Training accuracy: {:.2f}".format(acc_train))
    print("Testing accuracy: {:.2f}".format(acc_test))

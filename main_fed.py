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
from datetime import datetime

from utils.sampling import *
from utils.options import args_parser
from utils.optimization import *
from models.Update import *
from models.Nets import *
from models.Fed import *
from models.test import test_img
from AirComp import *
from sensing import *


if __name__ == '__main__':
    # set print args of numpy
    np.set_printoptions(edgeitems=6,threshold=1000,linewidth=200)

    # parse args
    args = args_parser()

    # load dataset and split users
    if args.dataset == 'mnist':
        # use cpu
        args.device = torch.device('cpu')
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
    elif args.dataset == 'cifar':
        # use gpu
        if torch.cuda.is_available():
            args.device = torch.device('cuda:0')
        elif torch.backends.mps.is_available():
            args.device = torch.device('mps')
        else:
            args.device = torch.device('cpu')
        trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])
        dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=trans_cifar)
    else:
        exit('Error: unrecognized dataset')

    if args.iid:
        dict_users = iid(dataset_train, args.num_users)
    else:
        dict_users = noniid(dataset_train, args.num_users)
    
    img_size = dataset_train[0][0].shape

    # build model
    if args.model == 'mobilenet':
        net_glob = MobileNet(args=args).to(args.device)
    elif args.model == 'resnet':
        net_glob = ResnetCifar(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'cifar':
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
    # neuron model settings
    D = sum(p.numel() for p in net_glob.parameters()) #number of model weights
    v_max = 5
    u_max = 0.5
    P_max = 0.1 * D
    beta_1 = 1.0
    beta_2 = 2.0

    # communication settings
    sigma_n = 1e-3 # sqrt(noise power)
    PL_exponent = 3.76 # User-BS Path loss exponent
    fc = 915 * 10**6 # carrier frequency 915MHz
    wave_lenth = 3.0 * 10**8 / fc # wave_lenth = c/f
    BS_gain = 10**(5.0/10) # BS antenna gain 5dBi
    User_gain = 10**(0.0/10) # user antenna gain 0dBi
    dist_max = 1000.0 # to quantify distance
    BS_hight = 10 # BS hight is 10m

    # optimal F is comming soon, now we just don't consider it
    F = [1.0 / np.sqrt(args.N)] * args.N

    if args.radius == 'same': 
        radius = np.array([25] * args.num_users) # same distance of all users
    elif args.radius == 'random_small':
        radius = np.random.rand(args.num_users) * 25 + 25 # (25,50)
    elif args.radius == 'random_large':
        radius = np.concatenate(np.random.rand(args.num_users-int(np.round(args.num_users/2))) * 25 + 25, int(np.round(args.num_users/2)) * 50 + 150) # half (25,50) and half (150,200)
    else:
        exit('Error: unrecognized radius')

    # get distances from users to BS
    BS_dist = np.sqrt(radius**2 + BS_hight**2)

    # get path loss
    Path_loss = BS_gain * User_gain * (wave_lenth / (4 * np.pi * BS_dist))**PL_exponent

    # get shadow loss
    #shadow_loss = (np.random.randn(args.N, args.num_users) + 1j * np.random.randn(args.N, args.num_users)) / 2**0.5
    shadow_loss = np.ones(args.N, args.num_users)

    print("PL", Path_loss)

    H_origin = shadow_loss * np.sqrt(Path_loss)

    print(H_origin)
    N, M = H_origin.shape
    noise = (np.random.randn(N, M)+1j*np.random.randn(N, M)) / 2**0.5 * sigma_n
    noise_factor = np.array(F).conj() @ noise #(1xN @ NxM = 1xM)
    inner = np.array(F).conj() @ H_origin #(1xN @ NxM = 1xM)
    p_v = np.sqrt(1)
    print("receive SNR:", (p_v * inner) / (v_max * noise_factor))
    # get the number of dataset per user have

    exit('test')
    data_per_user = []
    for _,v in dict_users.items():
        data_per_user.append(len(v))
    print(data_per_user)

    # derive optimal transmit power
    # not consider user selection M and beamforming vector f
    if args.all_clients:
        # all users are selected 
        P_u, P_v, P_G, user_list = optimal_power(P_max, beta_1, beta_2, data_per_user, F, H_origin, u_max, v_max, D, sigma_n)
    else:
        P_u, P_v, P_G, user_list = optimal_power_selection()

    H = H_origin[:, user_list]

    num_users = len(user_list)
    data_per_user_new = [data_per_user[idx] for idx in user_list]
    data_per_user = data_per_user_new
    total_size = sum(data_per_user)
    
    #print(total_size)
    
    for iter in range(args.epochs):
        loss_locals = []
        grad_locals = []
        mean_locals = []
        var_locals = []
        for idx in user_list:
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            grad, grad_mean, grad_var, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
            #train and collect results
            grad_locals.append(copy.deepcopy(grad))
            loss_locals.append(copy.deepcopy(loss))
            mean_locals.append(grad_mean)
            var_locals.append(grad_var)   

        # transfer list[grad1, grad2, ...,gradm] to matrix [m x D]
        grad_locals = np.array(grad_locals)
        mean_locals = np.array(mean_locals)
        var_locals = np.array(var_locals)
        print(grad_locals.shape)
        #print(grad_locals)          
        print("local gradients mean:   ", mean_locals) 
        print("local gradients variance", var_locals)                                                  
        
        # transmit and receive variances/ sign of means
        # var_receive: list [var1, var2, ..., varm]; sign_mean: list [+1, -1, -1, +1]
        var_receive, sign_mean = transmission_var(var_locals, mean_locals, P_v, F, H, v_max, sigma_n)

        # get transmit power of gradients and eta for AirComp
        P_g = P_G / D
        # formula (26),(27)
        # p_gm: np.array [p_g1, p_g2, ..., p_gm], eta: real factor
        p_gm, eta = get_args(var_receive, data_per_user, P_g, F, H)

        # transmit gradients with AirComp
        # signal_grad: matrix [m x D]; grad_locals: matrix [m x D]
        signal_grad = transmit_grad(grad_locals, mean_locals, var_locals, p_gm)

        # apply matched filtering to get distance
        # to be continue
        dist_locals = matched_filtering(signal_grad)
        
        # recieve gradients with AirComp
        # at this time, means have not been add to the grads
        # grad_receive: vector [1 x D]
        grad_receive = receive_grad(F, H, signal_grad, eta, sigma_n)

        # transmit and receive means / quantilized angle to transfer distances
        # mean_abs_receive: list [mean_abs1, mean_abs2, ..., mean_absm]; angle dist: list [angle1, angle2, ..., anglem]
        # dist_max: to quantilize angle
        mean_abs_receive, dist_receive = transmission_mean(mean_locals, dist_locals, P_u, F, H, u_max, dist_max, sigma_n)

        # formula (18): u = \sigma(Km * um)
        bias = np.array(data_per_user) @ (np.array(mean_abs_receive) * np.array(sign_mean))

        # add received means
        grad_receive = (grad_receive + bias) / total_size

        grad_groudtruth = np.average(grad_locals, axis=0, weights= np.array(data_per_user))

        error = grad_receive - grad_groudtruth

        print("Error of transmission:")
        print(error)
        print("ground_truth")
        print(grad_groudtruth)
        print("L2 norm of error: ", np.linalg.norm(error))

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
    plt.savefig('./save/fed_{}_{}_{}_iid{}_{}.png'.format(args.dataset, args.model, args.epochs, args.iid, datetime.now().strftime('%Y-%m-%d-%H:%M')))

    # testing
    net_glob.eval()
    acc_train, loss_train = test_img(net_glob, dataset_train, args)
    acc_test, loss_test = test_img(net_glob, dataset_test, args)
    print("Training accuracy: {:.2f}".format(acc_train))
    print("Testing accuracy: {:.2f}".format(acc_test))

    # save weight file
    torch.save(net_glob.state_dict(), './save/fed_{}_{}_{}_iid{}_{}.pth'.format(args.dataset, args.model, args.epochs, args.iid, datetime.now().strftime('%Y-%m-%d-%H:%M')))
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

from torch import nn
import torch.nn.functional as F
import torchvision


class MLP(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLP, self).__init__()
        self.layer_input = nn.Linear(dim_in, dim_hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.layer_hidden = nn.Linear(dim_hidden, dim_out)

    def forward(self, x):
        x = x.view(-1, x.shape[1]*x.shape[-2]*x.shape[-1])
        x = self.layer_input(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.layer_hidden(x)
        return x


class CNNMnist(nn.Module):
    def __init__(self, args):
        super(CNNMnist, self).__init__()
        self.conv1 = nn.Conv2d(args.num_channels, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, args.num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x


class CNNCifar(nn.Module):
    def __init__(self, args):
        super(CNNCifar, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, args.num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class ResnetCifar(nn.Module):
    def __init__(self, args):
        super(ResnetCifar, self).__init__()
        self.resnet = torchvision.models.resnet18(weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        # 7x7 kernel to 3x3 kernel
        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        # delete maxpool layer
        self.resnet.maxpool = nn.Identity()
        # change output dimension
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, args.num_classes)

    def forward(self, x):
        return self.resnet(x)
    
class MobileNetV3(nn.Module):
    def __init__(self, classes):
        super(MobileNetV3, self).__init__()
        self.mobilenet = torchvision.models.mobilenet_v3_large(weights = None)
        # do some change in mobilenet
        # reduce down-sampling times from 5 to 2
        # because mobilenet is proposed to classify (224x224) images
        # down-sampling from 224 to 224/2**5 = 7
        # we only down-sampling 2 times, from 32 to 32/2**2 = 8
        self.mobilenet.features[0][0].stride = (1,1)
        self.mobilenet.features[2].block[1][0].stride = (1,1)
        self.mobilenet.features[4].block[1][0].stride = (1,1)
        # modify classifier
        self.mobilenet.classifier[3] = nn.Linear(1280, classes)
        # reduce conv layers
        del self.mobilenet.features[1]
        del self.mobilenet.features[2]
        del self.mobilenet.features[3]

    def forward(self, x):
        return self.mobilenet(x)
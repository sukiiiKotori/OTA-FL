import torch
import os
from torch import nn
import torchvision
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import matplotlib.pyplot as plt

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
    
class MobileNetV3_small(nn.Module):
    def __init__(self, classes):
        super(MobileNetV3_small, self).__init__()
        self.mobilenet = torchvision.models.mobilenet_v3_small(weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        self.mobilenet.features[0][0].stride = (1,1)
        self.mobilenet.features[1].block[0][0].stride = (1,1)
        self.mobilenet.features[2].block[1][0].stride = (1,1)
        self.mobilenet.classifier[3] = nn.Linear(1024, classes)
        del self.mobilenet.features[3]
        del self.mobilenet.features[4]
        

    def forward(self, x):
        return self.mobilenet(x)

    
if __name__ == '__main__':
    net = MobileNetV3_small(10)
    print(sum(p.numel() for p in net.parameters()))
    t1=net
    print(t1)
    #trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])
    #dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_cifar)
    #dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=trans_cifar)
    #print(dataset_test[0][0].shape)
